

CUDA_VISIBLE_DEVICES=5 python tools/train_src.py --config-path st.dca.2urban

CUDA_VISIBLE_DEVICES=5 python tools/train_ssl_dca.py --config-path st.dca.2urban \
  --ckpt-model log/dca/2urban/src/Urban_best.pth \
  --gen 1

