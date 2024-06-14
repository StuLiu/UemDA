
export CUDA_VISIBLE_DEVICES=6

python tools/train_src.py --config-path st.uemda.pRgb2vaihingen \
  --align-domain 1

python tools/init_prototypes.py --config-path st.uemda.pRgb2vaihingen \
  --ckpt-model log/uemda/pRgb2vaihingen/src/Vaihingen_best.pth \
  --ckpt-proto log/uemda/pRgb2vaihingen/src/prototypes_best.pth \
  --stage 1

python tools/train_align_uem.py --config-path st.uemda.pRgb2vaihingen \
  --ckpt-model log/uemda/pRgb2vaihingen/src/Vaihingen_best.pth \
  --ckpt-proto log/uemda/pRgb2vaihingen/src/prototypes_best.pth \
  --align-domain 1 --gen 1 --refine-label 1

python tools/init_prototypes.py --config-path st.uemda.pRgb2vaihingen \
  --ckpt-model log/uemda/pRgb2vaihingen/align/Vaihingen_best.pth \
  --ckpt-proto log/uemda/pRgb2vaihingen/align/prototypes_best.pth \
  --stage 2

python tools/train_ssl_uem.py --config-path st.uemda.pRgb2vaihingen \
  --ckpt-model log/uemda/pRgb2vaihingen/align/Vaihingen_best.pth \
  --ckpt-proto log/uemda/pRgb2vaihingen/align/prototypes_best.pth \
  --gen 1 --refine-label 1 --lt uvem
