
CUDA_VISIBLE_DEVICES=2 python tools/train_src.py --config-path st.dca.2vaihingen

CUDA_VISIBLE_DEVICES=2 python tools/train_ssl_dca.py --config-path st.dca.2vaihingen \
  --ckpt-model log/dca/2vaihingen/src/Vaihingen_best.pth \
  --gen 1
