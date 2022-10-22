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

### 1.モデルにDeepSpeedの引数を適用する
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

### 2.DeepSpeedをinitializeする

```python

...

# 学習データの定義
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)

...

# モデルのインスタンス化
net = Net()

...

# netのパラメータのうち、誤差逆伝播を必要とするパラメータを抽出
parameters = filter(lambda p: p.requires_grad, net.parameters())

# モデルにDeepSpeedの引数を適用する
args=add_argument()

# DeepSpeedをinitalizeすることで以下の要素が使えるようにする
# 1) modelの分散処理
# 2) data loaderの分散処理
# 3) optimizerの分散処理
# 要するにZeROアルゴリズムを使うためにはこれらをする必要があるってことだと思う
model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=args, model=net, model_parameters=parameters, training_data=trainset)

```
※`filter()`関数は、与えられたシーケンスの各要素をフィルターする関数。第一引数に各要素に対して`True`か`False`を返す関数が、第二引数にはシーケンスが入る。

また、DeepSpeedのinitialize後は、元々あった`device`と`optimizer`は必要なくなるので、行を消すか、コメントアウトをする必要が出てくる。
```python
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net.to(device)

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

### 3. 学習API
`deepspeed.initialize`によって返された`model`は、訓練のために用いる*DeepSpeed Model Engine*である。
```python
for i, data in enumerate(trainloader):
    # get the inputs; data is a list of [inputs, labels]
    inputs = data[0].to(model_engine.device)
    labels = data[1].to(model_engine.device)

    outputs = model_engine(inputs)
    loss = critierion(outputs,labels)

    model_engine.backwards(loss)
    model_engine.step()
```
※ DeepSpeedでは、微分値の初期化は、パラメータ更新後に自動で行われる。

### 4. ds_config.json
次に、`deepspeed`のパラメータを保持した`ds_config.json`ファイルを書く。
```json
{
   "train_batch_size": 4,
   "steps_per_print": 2000,
   "optimizer": {
     "type": "Adam",
     "params": {
       "lr": 0.001,
       "betas": [
         0.8,
         0.999
       ],
       "eps": 1e-8,
       "weight_decay": 3e-7
     }
   },
   "scheduler": {
     "type": "WarmupLR",
     "params": {
       "warmup_min_lr": 0,
       "warmup_max_lr": 0.001,
       "warmup_num_steps": 1000
     }
   },
   "wall_clock_breakdown": false
 }
```


### 5. DeepSpeedを使ってCIFAR-10の学習を行う
以下のコマンドでDeepSpeedを使った学習ができる。DeepSpeedは**検知されたGPUを全て自動的に使う**。
```bash
deepspeed cifar10_deepspeed.py --deepspeed_config ds_config.json
```