import argparse
import numpy as np
import utils.train as train

from utils.data import SleepEDFDataset
from utils.augment import AugmentMethod
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Subset

# from transformers import BartModel, BartTokenizer, get_scheduler

def get_arges():
    """
    Parse command line arguments.
    
    Args:
        None.
    Returns:
        args: parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Pytorch vector quantizer constractive learning for Sleep Staging')
    # -----------------------------training parameters----------------------------- #
    parser.add_argument('--data', required=True, default="./dset/npz/EEG/train_data.npz", metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
    parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)',)
    parser.add_argument('--warmup', default=0.1, type=float,
                        help='warmup ratio for learning rate')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--workers', default=8, type=int,
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--log-every-n-steps', default=100, type=int,
                        help='Log every n steps')
    parser.add_argument('--gpu-index', default=0, type=int, 
                        help='Gpu index.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use for training.')     
    parser.add_argument('--ckpt', default=None, type=str,
                        help='Checkpoint to load.')
    parser.add_argument('--fp16precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')
    # -----------------------------pretrain parameters----------------------------- #
    parser.add_argument('--n_method', default=1, type=int,
                    help='number of methods to augment data')
    parser.add_argument('--pretrain_epochs', type=int,
                    help='number of pretrain epochs to run')
    parser.add_argument('--pretrain_lr', default=2e-5, type=float, metavar='pretrain lr',
                        help='initial pretrain learning rate')
    # -----------------------------finetune parameters----------------------------- #
    parser.add_argument('--finetune_epochs', type=int,
                        help='number of cls epochs to run')
    parser.add_argument('--finetune_lr', default=1e-3, type=float, metavar='finetune lr',
                        help='initial clssification learning rate')
    # -------------------------------model parameters------------------------------- #
    parser.add_argument('--in_channel', default=1, type=int, 
                        help='input channel form Sleep EDF data.')
    parser.add_argument('--h_dim', default=512, type=int, 
                        help='hidden dimension.')
    parser.add_argument('--vocab_size', default=10000, type=int, 
                        help='Number of embeddings.')
    parser.add_argument('--beta', default=0.25, type=float,
                        help='beta parameter for vector quantizer.')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout rate.')
    parser.add_argument('--n_labels', default=5, type=int, 
                        help='Number of labels.')
    parser.add_argument('--n_views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--pretrain', action='store_true',
                        help='pretrain the model.')
    parser.add_argument('--finetune', action='store_true',
                        help='Whether to train classification.')
    
    args = parser.parse_args()
    
    return args

def main():
    """
    Run the main training and evaluation loop.
    """
    args = get_arges()

    if args.pretrain:
        # augment data when contrastive learning pretraining
        augment = AugmentMethod(n_method=args.n_method)
        train_dataset = SleepEDFDataset(args.data, 
                                        augment=augment, )

        train_dataloader = DataLoader(train_dataset, 
                                    batch_size=args.batch_size // args.n_views,
                                    num_workers=args.workers,
                                    shuffle=True,)
        train.pretrain(train_dataloader, args)

    if args.finetune:
        dataset = SleepEDFDataset(args.data, )
        train_idx, val_idx= train_test_split(np.arange(len(dataset.labels)), 
                                        test_size=args.batch_size, 
                                        random_state=42, 
                                        shuffle=True, 
                                        stratify=dataset.labels)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_dataloader = DataLoader(train_dataset, 
                                    batch_size=args.batch_size,
                                    num_workers=args.workers,
                                    shuffle=True,)
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=args.batch_size,
                                    num_workers=args.workers,
                                    shuffle=False,)
        train.finetune(train_dataloader, val_dataloader, args)
if __name__ == '__main__':
    main()