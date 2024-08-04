import pandas as pd
import torch
import os
import pickle
import argparse

from preprocessing.model_select import chooseModel
from preprocessing.logging import Logger
from preprocessing.dataProcess import movieDataProcess, setup_seed

def tim_predict_dump(predict, test_data, out_path):
    predict_out = []
    movie_ids = test_data["movie_id"].tolist()
    user_ids = test_data["user_id"].tolist()
    for pred, movie_id, user_id in zip(predict, movie_ids, user_ids):
        # 1:user_dnn_embedding, 2:item_dnn_embedding
        # 3:target_recon_output_for_user, 4:non_target_recon_output_for_user
        # 5:target_recon_output_for_item, 6:non_target_recon_output_for_item
        predict_out.append([movie_id, user_id, pred])
    with open(out_path, 'wb') as file:
        pickle.dump(predict_out, file)

def main(args, log):
    # ["int_tower", "dssm",  "dat", "deep_fm", "dcn", "cold", "auto_int", "wide_and_deep", "tim_tower"]
    model_name = args.model_name
    ckpt_path = args.ckpt_path
    embedding_dim = args.embedding_dim
    setup_seed(seed=args.random_seed)

    assert os.path.exists(ckpt_path), f"Invalid checkpoint path {ckpt_path}"
    if args.dataset_name == "movieLens":
        dataSetProcess = movieDataProcess(log, args.data_path, embedding_dim)
    else:
        raise Exception(f"Dataset {args.dataset_name} not support")

    # 仅用于非双塔的模型
    linear_feature_columns = dataSetProcess.user_feature_columns + dataSetProcess.item_feature_columns
    user_feature_columns = dataSetProcess.user_feature_columns
    item_feature_columns = dataSetProcess.item_feature_columns
    dnn_feature_columns = linear_feature_columns

    model = chooseModel(model_name, user_feature_columns, item_feature_columns, linear_feature_columns,
                        dnn_feature_columns, dropout=args.dropout, device=args.device_name, log=log, data_name=args.dataset_name,
                        user_feature_columns_for_recon=dataSetProcess.user_feature_columns_for_recon,
                        item_feature_columns_for_recon=dataSetProcess.item_feature_columns_for_recon)
    
    model.load_state_dict(torch.load(ckpt_path))
    predict_result = model.predict(dataSetProcess.test_model_input, args.batch_size, True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.model_name + "." + args.dataset_name + ".pkl")
    if args.model_name == "tim":
        tim_predict_dump(predict_result, dataSetProcess.test, out_path)
    else:
        raise Exception(f"{args.model_name} is not support yet")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ["int_tower", "dssm",  "dat", "deep_fm", "dcn", "cold", "auto_int", "wide_and_deep", "tim", "kanTim"]
    parser.add_argument("--model_name", type=str, default="tim")
    parser.add_argument("--dataset_name", type=str, default="movieLens", help="[taobao|movieLens|amazon_eletronics]")
    parser.add_argument("--data_path", type=str, default="./data/movielens.txt")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/tim.ckpt")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device_name", type=str, default="cuda:0")
    parser.add_argument("--random_seed", type=int, default=1023)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default="pred")
    opt = parser.parse_args()
    log = Logger('./log/movielens_models.log', level='debug')
    main(opt, log)
