
## ResVAE PyTorch

VAE with residual blocks for AI dressroom project (Team <strong>IELD</strong>) 

## Datasets 

```bash
bash download.sh top-dataset
```


## Training networks
To train ResVAE

```bash
# celeba-hq
python main.py --mode train \
               --lambda_reg 1 \
               --train_img_dir data/top/train \
               --print_every 10 --save_every 200 --total_iters 10000\
               --batch_size 32 --lr 1e-4
```
