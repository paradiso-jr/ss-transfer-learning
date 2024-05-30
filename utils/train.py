import os
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.classifier import TimeSeriesBertClassifier
from utils.loss import info_nce_loss
from utils.functional import save_config_file, save_checkpoint, accuracy

from torch.cuda.amp import GradScaler, autocast
from transformers import get_scheduler
from tensorboardX import SummaryWriter

torch.autograd.set_detect_anomaly(True)
def pretrain(train_dataloader, args):

    # check if gpu training is available
    if args.device == 'cuda':
        device = torch.device(f'cuda:{args.gpu_index}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    model = TimeSeriesBertClassifier(in_channel=args.in_channel,
                                h_dim=args.h_dim,
                                vocab_size=args.vocab_size,
                                beta=args.beta,
                                n_labels=args.n_labels,)
        
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['state_dict'])
    
    # freeze fc classicier
    model.freeze_cls(enable=True)

    # enable AMP
    scaler = GradScaler(enabled=args.fp16precision)
    # load optimizer
    optimizer = optim.AdamW(model.parameters(), 
                            betas=(0.9, 0.9),
                            lr=args.pretrain_lr,
                            weight_decay=args.weight_decay,)
    
    # load scheduler
    num_training_steps = args.pretrain_epochs * len(train_dataloader)
    scheduler = get_scheduler('linear',
                            optimizer=optimizer,
                            num_warmup_steps=int(num_training_steps*args.warmup),
                            num_training_steps=num_training_steps
                            )
    model.to(device)
    model.train()

    n_iter = 0
    # tensorboard logger
    writer = SummaryWriter(comment="_contrastive_lerning_cnn_bert")
    save_config_file(writer.logdir, args)
    # logging
    logging.basicConfig(filename=os.path.join(writer.logdir, 'training.log'),    
                        level=logging.DEBUG)
    logging.info(f"Start contrastive training for {args.pretrain_epochs} epochs.")
    logging.info(f"Training with gpu: {device}.")
    logging.info(f"Training with # step: {len(train_dataloader)}.")
    logging.info(f"Training with batch_size: {args.batch_size}.")
    for epoch_counter in range(args.pretrain_epochs):
        for (xs, xs_aug), _ in train_dataloader:
            # forward pass
            xs_concat = torch.concat([xs, xs_aug], dim=0)
            xs_concat = xs_concat.unsqueeze(1).to(device)
            _, last_hidden_state, pooler_output,  embedding_loss, _ = model(xs_concat)

            logits, labels, nce_loss = info_nce_loss(pooler_output,
                                                n_views=args.n_views,
                                                temperature=args.temperature)
            
            loss = nce_loss + 5 * embedding_loss

            # AMP
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # logging
            if n_iter % args.log_every_n_steps == 0:
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                writer.add_scalar('loss', loss, global_step=n_iter)
                writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step=n_iter)
                # save checkpoint and config file
                checkpoint_name = "checkpoint_pretrain.pth.tar"
                save_checkpoint({
                    'epoch': args.pretrain_epochs,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(writer.logdir, checkpoint_name))
            n_iter += 1

        logging.info(f"Epoch {epoch_counter} loss: {loss.item()}\ttop1: {top1.item()}")
        logging.info(f"Epoch {epoch_counter} finished.")
    logging.info(f"Finished contrastive learning training.")
    logging.info(f"Model checkpoint and metadata has been saved at {writer.logdir}.")
    

def finetune(train_dataloader, val_dataloader, args):
    # check if gpu training is available
    if args.device == 'cuda':
        device = torch.device(f'cuda:{args.gpu_index}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    model = TimeSeriesBertClassifier(in_channel=args.in_channel,
                                h_dim=args.h_dim,
                                vocab_size=args.vocab_size,
                                beta=args.beta,
                                n_labels=args.n_labels,)
        
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['state_dict'])

    # freeze fc classicier
    model.freeze_encoder(enable=True)
    model.freeze_cls(enable=False)
    
    criterion = nn.CrossEntropyLoss()
    # enable AMP
    scaler = GradScaler(enabled=args.fp16precision)
    # load optimizer
    optimizer = optim.AdamW(model.parameters(), 
                            betas=(0.9, 0.9),
                            lr=args.finetune_lr,
                            weight_decay=args.weight_decay,)
    
    # learning scheduler
    num_training_steps = args.finetune_epochs * len(train_dataloader)
    scheduler = get_scheduler('cosine',
                            optimizer=optimizer,
                            num_warmup_steps=int(num_training_steps*args.warmup),
                            num_training_steps=num_training_steps
                            )

    model.to(device)
    
    n_iter = 0
    # tensorboard logger
    writer = SummaryWriter(comment="_finetune")
    save_config_file(writer.logdir, args)

    # logging
    logging.basicConfig(filename=os.path.join(writer.logdir, 'training.log'),    
                        level=logging.DEBUG)
    logging.info(f"Start finetune for {args.finetune_epochs} epochs.")
    logging.info(f"Training with gpu: {device}.")
    logging.info(f"Training with # step: {len(train_dataloader)}.")
    logging.info(f"Training with batch_size: {args.batch_size}.")

    for epoch_counter in range(args.finetune_epochs):
        for xs, labels in train_dataloader:
            # forward pass
            xs = xs.unsqueeze(1).to(device)
            labels = labels.long().to(device)
            logits, _, _,  _, _, = model(xs)
            loss = criterion(logits, labels)
            
            # AMP
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # logging
            if n_iter % args.log_every_n_steps == 0:
                model.eval()
                top1s = []
                top3s = []
                with torch.no_grad():
                    for xs_val, label_val in val_dataloader:
                        xs_val = xs_val.unsqueeze(1).to(device)
                        label_val = label_val.to(device)
                        logits, _, _,  _, _, = model(xs_val)
                        top1, top3 = accuracy(logits, label_val, topk=(1, 3))
                        top1s.append(top1[0].item())
                        top3s.append(top3[0].item())
                writer.add_scalar('loss', loss, global_step=n_iter)
                writer.add_scalar('acc/top1', sum(top1s) / len(top1s), global_step=n_iter)
                writer.add_scalar('acc/top3', sum(top3s) / len(top3s), global_step=n_iter)
                writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step=n_iter)
                # save checkpoint and config file
                checkpoint_name = "checkpoint_finetuned.pth.tar"
                save_checkpoint({
                    'epoch': args.finetune_epochs,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(writer.logdir, checkpoint_name))
                model.train()
            n_iter += 1

        logging.info(f"Epoch {epoch_counter} loss: {loss.item()}\ttop1: {top1.item()}")
        logging.info(f"Epoch {epoch_counter} finished.")
    logging.info(f"Finished fintune training.")
    logging.info(f"Model checkpoint and metadata has been saved at {writer.logdir}.")
