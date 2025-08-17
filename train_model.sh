# ["int_tower", "dssm",  "dat", "deep_fm", "dcn", "cold", "auto_int", "wide_and_deep", "tim"]
var="int_tower"
python train_movielens_whole_models.py --model_name $var > log/shell_log/$var.log 2>&1 &
#python train_movielens_whole_models.py --model_name "dssm" > log/train_model_sh.log 2>&1 &
#python train_movielens_whole_models.py --model_name "dat" > log/train_model_sh.log 2>&1 &
#python train_movielens_whole_models.py --model_name "deep_fm" > log/train_model_sh.log 2>&1 &
#python train_movielens_whole_models.py --model_name "dcn" > log/train_model_sh.log 2>&1 &
#python train_movielens_whole_models.py --model_name "cold" > log/train_model_sh.log 2>&1 &
#python train_movielens_whole_models.py --model_name "auto_int" > log/train_model_sh.log 2>&1 &
