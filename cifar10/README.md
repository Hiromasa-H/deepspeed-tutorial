# CIFAR10 tutorial

参考：[https://www.deepspeed.ai/tutorials/cifar-10/](https://www.deepspeed.ai/tutorials/cifar-10/)


## 通常のCIFAR10学習
```python
python3 cifar10_tutorial.py
```

```
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
100%|█████████████████████████████████████████████████████████████████████████| 170498071/170498071 [00:11<00:00, 14509349.19it/s]
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified
truck   cat truck  deer
[1,  2000] loss: 2.258
[1,  4000] loss: 1.921
[1,  6000] loss: 1.721
[1,  8000] loss: 1.609
[1, 10000] loss: 1.525
[1, 12000] loss: 1.479
[2,  2000] loss: 1.390
[2,  4000] loss: 1.380
[2,  6000] loss: 1.332
[2,  8000] loss: 1.323
[2, 10000] loss: 1.307
[2, 12000] loss: 1.301
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:    cat plane  ship plane
Accuracy of the network on the 10000 test images: 55 %
Accuracy of plane : 75 %
Accuracy of   car : 71 %
Accuracy of  bird : 35 %
Accuracy of   cat : 55 %
Accuracy of  deer : 52 %
Accuracy of   dog : 32 %
Accuracy of  frog : 55 %
Accuracy of horse : 60 %
Accuracy of  ship : 63 %
Accuracy of truck : 49 %
cuda:0
```

## DeepSpeed化

### モデルにDeepSpeedの引数を適用する
```python
import argparse
import deepspeed

def add_argument():

    parser=argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args=parser.parse_args()

    return args
```

### Initialization
`deepspeed.initialize`を使って、`model_engine`,`optimizer`,`trainlaoder`を作成する。
```python
def initialize(args,
               model,
               optimizer=None,
               model_params=None,
               training_data=None,
               lr_scheduler=None,
               mpu=None,
               dist_init_required=True,
               collate_fn=None):
```

