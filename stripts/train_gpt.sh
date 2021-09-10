#! /bin/bash


CUDA_VISIBLE_DEVICES=0 \
PYTHON_PATH=/home/inkoziev/polygon/ru-gpts \
python /home/inkoziev/polygon/ru-gpts/pretrain_transformers.py \
    --output_dir=../tmp/rugpt_model \
    --overwrite_output_dir \
    --model_type=gpt2 \
    --model_name_or_path='sberbank-ai/rugpt3small_based_on_gpt2' \
    --do_train \
    --line_by_line \
    --train_data_file=../data/gpt_dataset.txt \
    --per_gpu_train_batch_size 100 \
    --save_steps 3000000 \
    --block_size 512 \
    --num_train_epochs 5
