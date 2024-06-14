
export CUDA_VISIBLE_DEVICES=5

python tools/train_src.py --config-path st.proca.pRgb2potsdam

python tools/init_prototypes.py --config-path st.proca.pRgb2potsdam \
  --ckpt-model log/proca/pRgb2potsdam/src/Potsdam_best.pth \
  --ckpt-proto log/proca/pRgb2potsdam/src/prototypes_best.pth \
  --stage 1

python tools/train_align.py --config-path st.proca.pRgb2potsdam \
  --ckpt-model log/proca/pRgb2potsdam/src/Potsdam_best.pth \
  --ckpt-proto log/proca/pRgb2potsdam/src/prototypes_best.pth

python tools/init_prototypes.py --config-path st.proca.pRgb2potsdam \
  --ckpt-model log/proca/pRgb2potsdam/align/Potsdam_best.pth \
  --ckpt-proto log/proca/pRgb2potsdam/align/prototypes_best.pth \
  --stage 2

python tools/train_ssl.py --config-path st.proca.pRgb2potsdam \
  --ckpt-model log/proca/pRgb2potsdam/align/Potsdam_best.pth \
  --ckpt-proto log/proca/pRgb2potsdam/align/prototypes_best.pth \
  --gen 1
