import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training for calibration')

    # Datasets
    parser.add_argument('-d', '--dataset', default='ag_news', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed to use')
    parser.add_argument('--imbalance', default=0.02, type=float,
                        help='Imbalance to use in long tailed CIFAR10/100')
    parser.add_argument('--delta', default=0.25, type=float,
                        help='delta to use in Huber Loss in MDCA')
    # Optimization options
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                        help='batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--max-length', default=128, type=int,
                        metavar='MaxLength', help='max length of the input sequence')
 

    parser.add_argument('--alpha', default=5.0, type=float,
                        metavar='ALPHA', help='alpha to train Label Smoothing with')
    parser.add_argument('--beta', default=10, type=float,
                        metavar='BETA', help='beta to train DCA/MDCA with')
    parser.add_argument('--gamma', default=1, type=float,
                        metavar='GAMMA', help='gamma to train Focal Loss with')

    parser.add_argument('--drop', '--dropout', default=0, type=float,
                        metavar='Dropout', help='Dropout ratio')


    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--loss', default='cross_entropy', type=str, metavar='LNAME')
    parser.add_argument('--model', default='bert-base-uncased', type=str, metavar='MNAME')
    parser.add_argument("--num-labels", type=int, default=0, help="number of classes")
    # parser.add_argument('--optimizer', default='sgd', type=str, metavar='ONAME')

    # parser.add_argument('--prefix', default='', type=str, metavar='PRNAME')
    # parser.add_argument('--regularizer', default='l2', type=str, metavar='RNAME')
    # parser.add_argument('--patience', default=10, type=int)


    # Optimizer parameters
    # parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    # parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    # parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    # parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay (default: 0.0005)')


    # Distributed training
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--accum_iter", type=int, default=1, help='Gradient accumulation')

    
    # AdaFocal HPs
    parser.add_argument("--num-bins", type=int, default=15, dest="num_bins", help="Number of calibration bins")
    parser.add_argument("--adafocal-lambda", type=float, default=1.0, dest="adafocal_lambda", help="lambda for adafocal.")
    parser.add_argument("--adafocal-gamma-initial", type=float, default=1.0, dest="adafocal_gamma_initial", help="Initial gamma for each bin.")
    parser.add_argument("--adafocal-gamma-max", type=float, default=20.0, dest="adafocal_gamma_max", help="Maximum cutoff value for gamma.")
    parser.add_argument("--adafocal-gamma-min", type=float, default=-2.0, dest="adafocal_gamma_min", help="Minimum cutoff value for gamma.")
    parser.add_argument("--adafocal-switch-pt", type=float, default=0.2, dest="adafocal_switch_pt", help="Gamma at which to switch to inverse-focal loss.")
    parser.add_argument("--update-gamma-every", type=int, default=-1, dest="update_gamma_every", help="Update gamma every nth batch. If -1, update after epoch end.")
    



    return parser.parse_args()












