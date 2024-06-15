
export CUDA_VISIBLE_DEVICES=3
python tools/train_src.py --config-path st.dca.pRgb2potsdam

python tools/train_ssl_dca.py --config-path st.dca.pRgb2potsdam \
  --ckpt-model log/dca/pRgb2potsdam/src/Potsdam_best.pth \
  --gen 1

python tools/eval.py --config-path st.dca.pRgb2potsdam \
  --ckpt-path log/dca/pRgb2potsdam/ssl/Potsdam_best.pth \
  --test 1

