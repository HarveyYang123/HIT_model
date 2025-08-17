#!/bin/bash

#nohup sh run.sh >tmp.log 2>&1 &
arr=("int_tower" "dssm" "poly_encoder" "MVKE" "dat" "deep_fm" "dcn" "cold" "auto_int" "wide_and_deep" "hit" "kanTim")

# 定义目录路径
movielens_path="./log/movielens_shell_log"
rm -rf ./checkpoints/*
# 检查目录是否存在
if [ ! -d "$movielens_path" ]; then
  # 如果目录不存在，创建新的目录
  mkdir "$movielens_path"
  echo "目录已创建：$movielens_path"
else
  echo "目录已存在：$movielens_path"
fi

for value in ${arr[@]}; do
  echo "python train_movielens_whole_models.py --model_name $value --epoch 1000 > log/movielens_shell_log/$value.log 2>&1 &"
  python train_movielens_whole_models.py --model_name $value --epoch 1000 > log/movielens_shell_log/$value.log 2>&1 &
  pid1=$!

  wait $pid1
  echo "movielens model_name $value train complete"
done

# 定义目录路径
taobao_path="./log/taobao_shell_log"
rm -rf ./checkpoints/*
# 检查目录是否存在
if [ ! -d "$taobao_path" ]; then
  # 如果目录不存在，创建新的目录
  mkdir "$taobao_path"
  echo "目录已创建：$taobao_path"
else
  echo "目录已存在：$taobao_path"
fi

for value in ${arr[@]}; do
  echo "python train_taobao_whole_models.py --model_name $value --epoch 1000 > log/taobao_shell_log/$value.log 2>&1 &"
  python train_taobao_whole_models.py --model_name $value --epoch 1000 > log/taobao_shell_log/$value.log 2>&1 &
  pid1=$!

  wait $pid1
  echo "taobao model_name $value train complete"
done

# 定义目录路径
amazon_path="./log/amazon_shell_log"
rm -rf ./checkpoints/*
# 检查目录是否存在
if [ ! -d "$amazon_path" ]; then
  # 如果目录不存在，创建新的目录
  mkdir "$amazon_path"
  echo "目录已创建：$amazon_path"
else
  echo "目录已存在：$amazon_path"
fi

for value in ${arr[@]}; do
  echo "python trian_amazon_whole_models.py --model_name $value --epoch 1000 > log/amazon_shell_log/$value.log 2>&1 &"
  python trian_amazon_whole_models.py --model_name $value --epoch 1000 > log/amazon_shell_log/$value.log 2>&1 &
  pid1=$!

  wait $pid1
  echo "amazon model_name $value train complete"
done
