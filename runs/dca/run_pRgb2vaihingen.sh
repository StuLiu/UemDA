
export CUDA_VISIBLE_DEVICES=4
#python tools/train_src.py --config-path st.dca.pRgb2vaihingen

python tools/train_ssl_dca.py --config-path st.dca.pRgb2vaihingen \
  --ckpt-model log/dca/pRgb2vaihingen/src/Vaihingen_best.pth \
  --gen 1

python tools/eval.py --config-path st.dca.pRgb2vaihingen \
  --ckpt-path log/dca/pRgb2vaihingen/ssl/Vaihingen_best.pth \
  --test 1

