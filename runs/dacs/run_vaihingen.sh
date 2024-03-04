
CUDA_VISIBLE_DEVICES=4 python tools/train_src.py --config-path st.dacs.2vaihingen

CUDA_VISIBLE_DEVICES=4 python tools/train_ssl_mix.py --config-path st.dacs.2vaihingen \
  --ckpt-model log/dacs/2vaihingen/src/Vaihingen_best.pth \
  --gen 1 --mix classmix
