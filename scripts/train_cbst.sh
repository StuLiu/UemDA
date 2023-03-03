#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='st.cbst.2urban'
python CBST_train.py --config_path=${config_path}

# CUDA_VISIBLE_DEVICES=2 python CBST_train.py --config_path st.cbst.2urban