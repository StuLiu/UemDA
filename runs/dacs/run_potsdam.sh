
CUDA_VISIBLE_DEVICES=1 python tools/train_src.py --config-path st.dacs.2potsdam

CUDA_VISIBLE_DEVICES=1 python tools/train_ssl_mix.py --config-path st.dacs.2potsdam \
  --ckpt-model log/dacs/2potsdam/src/Potsdam_best.pth \
  --gen 1 --mix classmix
