
CUDA_VISIBLE_DEVICES=3 python tools/train_src.py --config-path st.cutmix.2potsdam

CUDA_VISIBLE_DEVICES=3 python tools/train_ssl_mix.py --config-path st.cutmix.2potsdam \
  --ckpt-model log/cutmix/2potsdam/src/Potsdam_best.pth \
  --gen 1 --mix cutmix
