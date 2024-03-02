

CUDA_VISIBLE_DEVICES=4 python tools/train_src.py --config-path st.dca.2rural

CUDA_VISIBLE_DEVICES=4 python tools/train_ssl_dca.py --config-path st.dca.2rural \
  --ckpt-model log/dca/2rural/src/Rural_best.pth \
  --gen 1

