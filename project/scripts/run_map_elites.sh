#!/bin/bash

output_dir="/home/andresf/workspace/CPSC-532J-NeverEndingRL/results/"

for i in 1 2 3 4 5
do

   echo "Run $i"

   python -u code/lunar_lander_map_elites_complex_model.py \
       --output_dir "${output_dir}" \
       --num_generations 1000 \
       --seed 0

done
