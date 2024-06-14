
export CUDA_VISIBLE_DEVICES=7

python tools/train_src.py --config-path st.cutmix.pRgb2potsdam

python tools/train_ssl_mix.py --config-path st.cutmix.pRgb2potsdam \
  --ckpt-model log/cutmix/pRgb2potsdam/src/Potsdam_best.pth \
  --gen 1 --mix cutmix

python tools/train_src.py --config-path st.cutmix.pRgb2vaihingen

python tools/train_ssl_mix.py --config-path st.cutmix.pRgb2vaihingen \
  --ckpt-model log/cutmix/pRgb2vaihingen/src/Vaihingen_best.pth \
  --gen 1 --mix cutmix
