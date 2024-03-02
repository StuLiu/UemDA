
CUDA_VISIBLE_DEVICES=3 python tools/train_src.py --config-path st.dca.2potsdam

CUDA_VISIBLE_DEVICES=3 python tools/train_ssl_dca.py --config-path st.dca.2potsdam \
  --ckpt-model log/dca/2potsdam/src/Potsdam_best.pth \
  --gen 1

