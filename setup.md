# 環境構築


参考：[https://www.deepspeed.ai/tutorials/advanced-install/](https://www.deepspeed.ai/tutorials/advanced-install/)

インストール
```
pyenv virtualenv 3.8.5 deepspseed
pyenv local deepspeed
pip install deepspseed
```

合うバージョンのtorchをインストール
```
pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
```

インストールができたことの確認
```
ds_report
```

```
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [OKAY]
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
cpu_adam ............... [NO] ....... [OKAY]
cpu_adagrad ............ [NO] ....... [OKAY]
fused_adam ............. [NO] ....... [OKAY]
fused_lamb ............. [NO] ....... [OKAY]
 [WARNING]  please install triton==1.0.0 if you want to use sparse attention
sparse_attn ............ [NO] ....... [NO]
transformer ............ [NO] ....... [OKAY]
stochastic_transformer . [NO] ....... [OKAY]
 [WARNING]  async_io requires the dev libaio .so object and headers but these were not found.
 [WARNING]  async_io: please install the libaio-dev package with apt
 [WARNING]  If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
async_io ............... [NO] ....... [NO]
utils .................. [NO] ....... [OKAY]
quantizer .............. [NO] ....... [OKAY]
transformer_inference .. [NO] ....... [OKAY]
--------------------------------------------------
DeepSpeed general environment info:
torch install path ............... ['/home/hhiromasa/.pyenv/versions/3.8.5/envs/deepspeed/lib/python3.8/site-packages/torch']
torch version .................... 1.12.1+cu102
torch cuda version ............... 10.2
torch hip version ................ None
nvcc version ..................... 10.0
deepspeed install path ........... ['/home/hhiromasa/.pyenv/versions/3.8.5/envs/deepspeed/lib/python3.8/site-packages/deepspeed']
deepspeed info ................... 0.7.4, unknown, unknown
deepspeed wheel compiled w. ...... torch 1.12, cuda 10.2
```


