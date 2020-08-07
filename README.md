
## ResVAE PyTorch

VAE with residual blocks for AI dressroom project (Team <strong>IELD</strong>) 

## Datasets 

```bash
bash download.sh top_img
bash download.sh bottom_img
```


## Training networks
To train ResVAE

```bash
# celeba-hq
python main.py --mode train \
               --lambda_reg 1 \
               --train_img_dir data/top_img \
               --print_every 5 --save_every 10 --total_iters 1000\
               --batch_size 16 --lr 1e-4
```
