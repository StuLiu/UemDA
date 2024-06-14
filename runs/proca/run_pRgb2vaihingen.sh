
export CUDA_VISIBLE_DEVICES=6

python tools/train_src.py --config-path st.proca.pRgb2vaihingen

python tools/init_prototypes.py --config-path st.proca.pRgb2vaihingen \
  --ckpt-model log/proca/pRgb2vaihingen/src/Vaihingen_best.pth \
  --ckpt-proto log/proca/pRgb2vaihingen/src/prototypes_best.pth \
  --stage 1

python tools/train_align.py --config-path st.proca.pRgb2vaihingen \
  --ckpt-model log/proca/pRgb2vaihingen/src/Vaihingen_best.pth \
  --ckpt-proto log/proca/pRgb2vaihingen/src/prototypes_best.pth

python tools/init_prototypes.py --config-path st.proca.pRgb2vaihingen \
  --ckpt-model log/proca/pRgb2vaihingen/align/Vaihingen_best.pth \
  --ckpt-proto log/proca/pRgb2vaihingen/align/prototypes_best.pth \
  --stage 2

python tools/train_ssl.py --config-path st.proca.pRgb2vaihingen \
  --ckpt-model log/proca/pRgb2vaihingen/align/Vaihingen_best.pth \
  --ckpt-proto log/proca/pRgb2vaihingen/align/prototypes_best.pth \
  --gen 1
