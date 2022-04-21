#!/bin/bash
python run_spanqa.py \
    --model_path /home/lsz/models/google-electra-base-discriminator \
    --data_dir ./data/ \
    --learning_rate 5e-5 \
    --num_train_epochs 3
