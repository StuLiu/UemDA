
CUDA_VISIBLE_DEVICES=4 python tools/train_src.py --config-path st.cutmix.2urban

CUDA_VISIBLE_DEVICES=4 python tools/train_ssl_mix.py --config-path st.cutmix.2urban \
  --ckpt-model log/cutmix/2urban/src/Urban_best.pth \
  --gen 1 --mix cutmix
