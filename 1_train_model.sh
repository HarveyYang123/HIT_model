#!/bin/bash

# ["int_tower", "dssm",  "dat", "deep_fm", "dcn", "cold", "auto_int", "wide_and_deep", "tim"]
#models=("int_tower" "dssm" "dat" "deep_fm" "dcn" "cold" "auto_int" "wide_and_deep" "tim")
models=("dat" "deep_fm" "dssm" "int_tower" "wide_and_deep")
for model in "${models[@]}"; do
  echo "Current model: $model"
  # 在这里执行你需要对每个model变量进行的操作
  python train_movielens_whole_models.py --model_name "$model" > log/shell_log/$model.log 2>&1 &
done

