
CUDA_VISIBLE_DEVICES=5 python tools/train_src.py --config-path st.cutmix.2rural

CUDA_VISIBLE_DEVICES=5 python tools/train_ssl_mix.py --config-path st.cutmix.2rural \
  --ckpt-model log/cutmix/2rural/src/Rural_best.pth \
  --gen 1 --mix cutmix
