import argparse
import deepspeed

# use `deepspeed.add_config_arguments` to add DeepSpeed configuration arguments
def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    # data
    # cuda
    parser.add_argument('--with_cuda', default=False, action='store_true',help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',help='use exponential moving average')

    # train
    parser.add_argument('-b','--batch_size', default=128, type=int, help='batch size (default: 128)')
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')

    # Include DeepSpeed confiuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args