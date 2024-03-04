
CUDA_VISIBLE_DEVICES=2 python tools/train_src.py --config-path st.dacs.2rural

CUDA_VISIBLE_DEVICES=2 python tools/train_ssl_mix.py --config-path st.dacs.2rural \
  --ckpt-model log/dacs/2rural/src/Rural_best.pth \
  --gen 1 --mix classmix
