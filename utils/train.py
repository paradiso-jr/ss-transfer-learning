import os
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from datetime import datetime
from torchsampler import ImbalancedDatasetSampler

from models.classifier import TimeSeriesBertClassifier
from utils.loss import info_nce_loss
from utils.functional import save_config_file, save_checkpoint, accuracy, calculate_accuracy_per_label
from torchmetrics import CohenKappa, F1Score
from torch.cuda.amp import GradScaler, autocast
from transformers import get_scheduler, get_polynomial_decay_schedule_with_warmup
from tensorboardX import SummaryWriter

def warmup(current_step: int):
    if current_step < args.warmup_steps:  # current_step / warmup_steps * base_lr
        return float(current_step / args.warmup_steps)
    else:                                 # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
        return max(0.0, float(args.training_steps - current_step) / float(max(1, args.training_steps - args.warmup_steps)))
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
                            betas=(0.99, 0.9),
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
            
            loss = nce_loss + embedding_loss

            # AMP
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
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
    #model.freeze_encoder(enable=False)
    #model.freeze_cls(enable=False)
    #model.freeze_bert(enable=True)
    
    pos_weight = torch.tensor([1.8549,  8.2645,  1.8836, 14.2857,  6.7159]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # enable AMP
    scaler = GradScaler(enabled=args.fp16precision)
    # load optimizer
    optimizer = optim.AdamW(model.parameters(), 
                            betas=(0.99, 0.9),
                            lr=args.finetune_lr,
                            weight_decay=args.weight_decay,)
    
    # learning scheduler
    num_training_steps = args.finetune_epochs * len(train_dataloader)
    
    '''
    scheduler = get_scheduler('cosine',
                            optimizer=optimizer,
                            num_warmup_steps=int(num_training_steps*args.warmup),
                            num_training_steps=num_training_steps
                            )
    '''
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=int(num_training_steps*args.warmup), 
                                                    num_training_steps=num_training_steps)

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
            try:
                # forward pass
                xs = xs.unsqueeze(1).to(device)
                labels = labels.long().to(device)
                labels = F.one_hot(labels, num_classes=args.n_labels).float()
                logits, _, _,  emb_loss, _, = model(xs)
                logits = torch.sigmoid(logits)
                loss = criterion(logits, labels)
                loss = loss
                # AMP
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                # logging
                if n_iter % args.log_every_n_steps == 0:
                    model.eval()
                    top1s = []
                    top3s = []
                    each_labels = []
                    with torch.no_grad():
                        for xs_val, label_val in val_dataloader:
                            xs_val = xs_val.unsqueeze(1).to(device)
                            label_val = label_val.to(device)
                            logits, _, _,  _, _, = model(xs_val)
                            predicted_labels = torch.argmax(logits, dim=1)
                            accuracies = calculate_accuracy_per_label(predicted_labels, label_val)
                            
                            top1, top3 = accuracy(logits, label_val, topk=(1, 3))
                            each_labels.append([x.item() for x in accuracies])
                            top1s.append(top1[0].item())
                            top3s.append(top3[0].item())
                    
                    writer.add_scalar('loss', loss, global_step=n_iter)
                    writer.add_scalar('acc/top1', sum(top1s) / len(top1s), global_step=n_iter)
                    writer.add_scalar('acc/top3', sum(top3s) / len(top3s), global_step=n_iter)
                    writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step=n_iter)
                    # save checkpoint and config file
                    each_labels = np.array(each_labels)
                    each_labels = np.mean(each_labels, axis=0)
                    for i, accuarcy in enumerate(each_labels):
                        writer.add_scalar(f'acc_labels/class_{i}_accuary', accuarcy, global_step=n_iter)

                    checkpoint_name = "checkpoint_finetuned.pth.tar"
                    save_checkpoint({
                        'epoch': args.finetune_epochs,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, is_best=False, filename=os.path.join(writer.logdir, checkpoint_name))
                    model.train()
                n_iter += 1
            except Exception as e:
                sys.stderr.write(str(e)+'\n')
                continue

        logging.info(f"Epoch {epoch_counter} loss: {loss.item()}\ttop1: {top1.item()}")
        logging.info(f"Epoch {epoch_counter} finished.")

    logging.info(f"Finished fintune training.")
    logging.info(f"Model checkpoint and metadata has been saved at {writer.logdir}.")

def evaluate(test_dataloader, args):
    # check if gpu training is available
    
    if args.device == 'cuda':
        device = torch.device(args.gpu_index)
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print(device)
    cohenkappa = CohenKappa(task="multiclass", num_classes=args.n_labels).to(device)
    f1 = F1Score(task="multiclass", num_classes=args.n_labels).to(device)
    model = TimeSeriesBertClassifier(in_channel=args.in_channel,
                                h_dim=args.h_dim,
                                vocab_size=args.vocab_size,
                                beta=args.beta,
                                n_labels=args.n_labels,)
    model.to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    
    
    model.eval()

    # Get the current time
    current_time = datetime.now()

    # Format the current time for the filename (avoiding colons and microseconds for compatibility)
    filename_time = current_time.strftime("%Y%m%d_%H%M%S")

    # Define the filename with the timestamp
    filename = f"evaluation_{filename_time}.log"
    # logging
    logging.basicConfig(filename=os.path.join("./runs", filename),    
                        level=logging.DEBUG)
    logging.info(f"Testing with # step: {len(test_dataloader)}.")
    logging.info(f"Testing with dataset: {args.data}.")
    logging.info(f"Testing on the testing set...")
    top1s = []
    top3s = []
    cohenkappas = []
    f1s = []
    each_labels = []
    with torch.no_grad():
        for xs_val, label_val in test_dataloader:
            xs_val = xs_val.unsqueeze(1).to(device)
            label_val = label_val.to(device)
            logits, _, _,  _, _, = model(xs_val)
            predicted_labels = torch.argmax(logits, dim=1)
            accuracies = calculate_accuracy_per_label(predicted_labels, label_val)
            top1, top3 = accuracy(logits, label_val, topk=(1, 3))

            each_labels.append([x.item() for x in accuracies])
            top1s.append(top1[0].item())
            top3s.append(top3[0].item())
            cohenkappas.append(cohenkappa(predicted_labels, label_val).item())
            f1s.append(f1(predicted_labels, label_val).item())
    logging.info("Accuracy per class: ")
    for i, acc in enumerate(np.mean(each_labels, axis=0)):
        logging.info(f'\t\t Class {i}: \t Accuracy: {acc:.4%}.')
    logging.info(f"Top-1 accuracy: {sum(top1s)/len(top1s)}.")
    logging.info(f"Top-3 accuracy: {sum(top3s)/len(top3s)}.")
    logging.info(f"Cohen Kappa score: {sum(cohenkappas)/len(cohenkappas)}.")
    logging.info(f"F1 Score: {sum(f1s)/len(f1s)}.")