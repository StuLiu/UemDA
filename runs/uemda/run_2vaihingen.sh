
CUDA_VISIBLE_DEVICES=6 python tools/train_src.py --config-path st.uemda.2vaihingen

CUDA_VISIBLE_DEVICES=6 python tools/init_prototypes.py --config-path st.uemda.2vaihingen \
  --ckpt-model log/uemda/2vaihingen/src/Vaihingen_best.pth \
  --ckpt-proto log/uemda/2vaihingen/src/prototypes_best.pth \
  --stage 1

CUDA_VISIBLE_DEVICES=6 python tools/train_align_uem.py --config-path st.uemda.2vaihingen \
  --ckpt-model log/uemda/2vaihingen/src/Vaihingen_best.pth \
  --ckpt-proto log/uemda/2vaihingen/src/prototypes_best.pth \
  --align-domain 1 --gen 1

CUDA_VISIBLE_DEVICES=6 python tools/init_prototypes.py --config-path st.uemda.2vaihingen \
  --ckpt-model log/uemda/2vaihingen/align/Vaihingen_best.pth \
  --ckpt-proto log/uemda/2vaihingen/align/prototypes_best.pth \
  --stage 2

CUDA_VISIBLE_DEVICES=6 python tools/train_ssl_uem.py --config-path st.uemda.2vaihingen \
  --ckpt-model log/uemda/2vaihingen/align/Vaihingen_best.pth \
  --ckpt-proto log/uemda/2vaihingen/align/prototypes_best.pth \
  --gen 1 --refine-label 1 --lt uvem
