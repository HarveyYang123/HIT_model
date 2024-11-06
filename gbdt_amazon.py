
import numpy as np
import random
import os
import argparse

import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from preprocessing.logging import Logger
from preprocessing.dataProcess import amazonDataProcess, setup_seed
from preprocessing.inputs import SparseFeat, DenseFeat, build_input_features


def input_from_feature_columns(X, feature_columns):
    feature_index = build_input_features(feature_columns)
    length = 0
    for name, _ in feature_index.items():
        if length == 0:
            length = len(X[name])
        else:
            if length > len(X[name]):
                length = len(X[name])

    dense_value_list = []
    for i in range(length):
        dense_list = []
        for name, _ in feature_index.items():
            # print(f"type:{type(X[name])}")
            # print(f"name:{name}; X[name]:{X[name].tolist()[:10]}")
            # print("================")
            x = X[name].tolist()[i]
            if isinstance(x, np.ndarray):
                for v in x:
                    dense_list.append(v)
            else:
                dense_list.append(x)

        dense_value_list.append(dense_list)

    print(f"dense_value_list:{dense_value_list[0]}")
    return dense_value_list

def main(args, log):
    embedding_dim = args.embedding_dim
    setup_seed(seed=args.random_seed)
    AmazonData = amazonDataProcess(log, args.data_path, embedding_dim)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.1,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }

    linear_feature_columns = AmazonData.user_feature_columns + AmazonData.item_feature_columns

    train_input = input_from_feature_columns(AmazonData.train_model_input, linear_feature_columns)
    dtrain = xgb.DMatrix(train_input, label=AmazonData.train[AmazonData.target].values)

    test_input = input_from_feature_columns(AmazonData.test_model_input, linear_feature_columns)

    dtest = xgb.DMatrix(test_input)

    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds)

    # print(f"dtest:{dtest}")
    y_pred = model.predict(dtest)
    y_pred_binary = np.round(y_pred)
    label = [v[0] for v in AmazonData.test[AmazonData.target].values]

    log.logger.info(f"test AUC, {round(roc_auc_score(label, y_pred_binary), 4)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/amazon_eletronics.csv")
    # parser.add_argument("--data_path", type=str, default="./data/movielens_test.txt")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=1023)
    opt = parser.parse_args()
    log = Logger('./log/movielens_data.log', level='debug')
    main(opt, log)