
CUDA_VISIBLE_DEVICES=3 python tools/train_src.py --config-path st.dacs.2urban

CUDA_VISIBLE_DEVICES=3 python tools/train_ssl_mix.py --config-path st.dacs.2urban \
  --ckpt-model log/dacs/2urban/src/Urban_best.pth \
  --gen 1 --mix classmix
