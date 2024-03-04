
CUDA_VISIBLE_DEVICES=2 python tools/train_src.py --config-path st.cutmix.2vaihingen

CUDA_VISIBLE_DEVICES=2 python tools/train_ssl_mix.py --config-path st.cutmix.2vaihingen \
  --ckpt-model log/cutmix/2vaihingen/src/Vaihingen_best.pth \
  --gen 1 --mix cutmix
