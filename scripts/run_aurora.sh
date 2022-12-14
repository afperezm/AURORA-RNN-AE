#!/bin/bash

output_dir="/home/andresf/workspace/CPSC-532J-NeverEndingRL/results/"

for i in 1 2 3 4 5; do

  echo "Run $i"

  python -u code/lunar_lander_aurora_complex_model.py \
    --output_dir "${output_dir}" \
    --train_mode "pre" \
    --encode_mode "ae" \
    --time_mode "stack" \
    --num_generations 1000 \
    --seed 0

  python -u code/lunar_lander_aurora_complex_model.py \
    --output_dir "${output_dir}" \
    --train_mode "inc" \
    --encode_mode "ae" \
    --time_mode "stack" \
    --num_generations 1000 \
    --seed 0

  python -u code/lunar_lander_aurora_complex_model.py \
    --output_dir "${output_dir}" \
    --train_mode "pre" \
    --encode_mode "vae" \
    --time_mode "stack" \
    --num_generations 1000 \
    --seed 0

  python -u code/lunar_lander_aurora_complex_model.py \
    --output_dir "${output_dir}" \
    --train_mode "inc" \
    --encode_mode "vae" \
    --time_mode "stack" \
    --num_generations 1000 \
    --seed 0

  python -u code/lunar_lander_aurora_complex_model.py \
    --output_dir "${output_dir}" \
    --train_mode "pre" \
    --encode_mode "rnn-ae" \
    --time_mode "pad" \
    --num_generations 1000 \
    --learning_rate 0.01 \
    --seed 0

  python -u code/lunar_lander_aurora_complex_model.py \
    --output_dir "${output_dir}" \
    --train_mode "inc" \
    --encode_mode "rnn-ae" \
    --time_mode "pad" \
    --num_generations 1000 \
    --learning_rate 0.01 \
    --seed 0

done
