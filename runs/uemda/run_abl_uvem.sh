
export CUDA_VISIBLE_DEVICES=5

python tools/train_ssl_uvem-abl.py --config-path st.uemda.2vaihingen \
  --ckpt-model log/uemda/2vaihingen/src/Vaihingen_best.pth \
  --ckpt-proto log/uemda/2vaihingen/src/prototypes_best.pth \
  --gen 1 --refine-label 0 --lt uvem --uvem-m 0.2 --uvem-g 0.5

python tools/train_ssl_uvem-abl.py --config-path st.uemda.2vaihingen \
  --ckpt-model log/uemda/2vaihingen/src/Vaihingen_best.pth \
  --ckpt-proto log/uemda/2vaihingen/src/prototypes_best.pth \
  --gen 1 --refine-label 0 --lt uvem --uvem-m 0.2 --uvem-g 1.0

python tools/train_ssl_uvem-abl.py --config-path st.uemda.2vaihingen \
  --ckpt-model log/uemda/2vaihingen/src/Vaihingen_best.pth \
  --ckpt-proto log/uemda/2vaihingen/src/prototypes_best.pth \
  --gen 1 --refine-label 0 --lt uvem --uvem-m 0.2 --uvem-g 2.0

python tools/train_ssl_uvem-abl.py --config-path st.uemda.2vaihingen \
  --ckpt-model log/uemda/2vaihingen/src/Vaihingen_best.pth \
  --ckpt-proto log/uemda/2vaihingen/src/prototypes_best.pth \
  --gen 1 --refine-label 0 --lt uvem --uvem-m 0.2 --uvem-g 4.0

python tools/train_ssl_uvem-abl.py --config-path st.uemda.2vaihingen \
  --ckpt-model log/uemda/2vaihingen/src/Vaihingen_best.pth \
  --ckpt-proto log/uemda/2vaihingen/src/prototypes_best.pth \
  --gen 1 --refine-label 0 --lt uvem --uvem-m 0.2 --uvem-g 8.0

python tools/train_ssl_uvem-abl.py --config-path st.uemda.2vaihingen \
  --ckpt-model log/uemda/2vaihingen/src/Vaihingen_best.pth \
  --ckpt-proto log/uemda/2vaihingen/src/prototypes_best.pth \
  --gen 1 --refine-label 0 --lt uvem --uvem-m 0.2 --uvem-g 16.0

python tools/train_ssl_uvem-abl.py --config-path st.uemda.2vaihingen \
  --ckpt-model log/uemda/2vaihingen/src/Vaihingen_best.pth \
  --ckpt-proto log/uemda/2vaihingen/src/prototypes_best.pth \
  --gen 1 --refine-label 0 --lt uvem --uvem-m 0.2 --uvem-g 32.0