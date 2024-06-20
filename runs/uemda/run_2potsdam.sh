
export CUDA_VISIBLE_DEVICES=7

python tools/train_src.py --config-path st.uemda.2potsdam \
  --align-domain 1

python tools/init_prototypes.py --config-path st.uemda.2potsdam \
  --ckpt-model log/uemda/2potsdam/src/Potsdam_best.pth \
  --ckpt-proto log/uemda/2potsdam/src/prototypes_best.pth \
  --stage 1

python tools/train_align_uem.py --config-path st.uemda.2potsdam \
  --ckpt-model log/uemda/2potsdam/src/Potsdam_best.pth \
  --ckpt-proto log/uemda/2potsdam/src/prototypes_best.pth \
  --align-domain 1 --gen 1 --refine-label 1

python tools/init_prototypes.py --config-path st.uemda.2potsdam \
  --ckpt-model log/uemda/2potsdam/align/Potsdam_best.pth \
  --ckpt-proto log/uemda/2potsdam/align/prototypes_best.pth \
  --stage 2

python tools/train_ssl_uem.py --config-path st.uemda.2potsdam \
  --ckpt-model log/uemda/2potsdam/align/Potsdam_best.pth \
  --ckpt-proto log/uemda/2potsdam/align/prototypes_best.pth \
  --gen 1 --refine-label 1 --lt uvem
