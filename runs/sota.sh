

CUDA_VISIBLE_DEVICES=4 python GAST_train_pseudo.py --config-path st.gast.2potsdam   --refine-label 1 --refine-mode all --ls OhemCrossEntropy --bcs 1 --lt uvem --bct 1 --class-temp 2.0
CUDA_VISIBLE_DEVICES=5 python GAST_train_pseudo.py --config-path st.gast.2vaihingen --refine-label 1 --refine-mode all --ls CrossEntropy --bcs 0 --lt uvem --bct 1 --class-temp 2.0