#!/bin/bash
arr=("int_tower" "dssm"  "dat" "deep_fm" "dcn" "cold" "auto_int" "wide_and_deep" "tim" "kan_Tim")

# 定义目录路径
movielens_path="./log/movielens_shell_log"

# 检查目录是否存在
if [ ! -d "$movielens_path" ]; then
  # 如果目录不存在，创建新的目录
  mkdir "$movielens_path"
  echo "目录已创建：$movielens_path"
else
  echo "目录已存在：$movielens_path"
fi

for value in ${arr[@]}; do
  echo "movielens model_name：$value"
  python train_movielens_whole_models.py --model_name $value > log/movielens_shell_log/$value.log 2>&1 &
  pid1=$!

  wait $pid1
  echo "movielens model_name $value train complete"
done

# 定义目录路径
taobao_path="./log/taobao_shell_log"

# 检查目录是否存在
if [ ! -d "$taobao_path" ]; then
  # 如果目录不存在，创建新的目录
  mkdir "$taobao_path"
  echo "目录已创建：$taobao_path"
else
  echo "目录已存在：$taobao_path"
fi

for value in ${arr[@]}; do
  echo "taobao model_name：$value"
  python trian_taobao_whole_models.py --model_name $value > log/taobao_shell_log/$value.log 2>&1 &
  pid1=$!

  wait $pid1
  echo "taobao model_name $value train complete"
done

# 定义目录路径
amazon_path="./log/amazon_shell_log"

# 检查目录是否存在
if [ ! -d "$amazon_path" ]; then
  # 如果目录不存在，创建新的目录
  mkdir "$amazon_path"
  echo "目录已创建：$amazon_path"
else
  echo "目录已存在：$amazon_path"
fi

for value in ${arr[@]}; do
  echo "amazon model_name：$value"
  python trian_amazon_whole_models.py --model_name $value > log/amazon_shell_log/$value.log 2>&1 &
  pid1=$!

  wait $pid1
  echo "amazon model_name $value train complete"
done
