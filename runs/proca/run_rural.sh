
CUDA_VISIBLE_DEVICES=7 python tools/train_src.py --config-path st.proca.2rural

CUDA_VISIBLE_DEVICES=7 python tools/init_prototypes.py --config-path st.proca.2rural \
  --ckpt-model log/proca/2rural/src/Rural_best.pth \
  --ckpt-proto log/proca/2rural/src/prototypes_best.pth \
  --stage 1

CUDA_VISIBLE_DEVICES=7 python tools/train_align.py --config-path st.proca.2rural \
  --ckpt-model log/proca/2rural/src/Rural_best.pth \
  --ckpt-proto log/proca/2rural/src/prototypes_best.pth \
  --align-domain 0

CUDA_VISIBLE_DEVICES=7 python tools/init_prototypes.py --config-path st.proca.2rural \
  --ckpt-model log/proca/2rural/align/Rural_best.pth \
  --ckpt-proto log/proca/2rural/align/prototypes_best.pth \
  --stage 2

CUDA_VISIBLE_DEVICES=7 python tools/train_ssl.py --config-path st.proca.2rural \
  --ckpt-model log/proca/2rural/align/Rural_best.pth \
  --ckpt-proto log/proca/2rural/align/prototypes_best.pth \
  --gen 1 --refine-label 0 --lt none
