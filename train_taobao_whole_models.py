import numpy as np
import pandas as pd
import torch
import random
import os
import argparse

from sklearn.metrics import log_loss, roc_auc_score
from preprocessing.model_select import chooseModel
from preprocessing.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing.logging import Logger
from preprocessing.dataProcess import taobaoDataProcess, setup_seed


def main(args, log):
    model_name = args.model_name
    ckpt_fold = args.ckpt_fold

    embedding_dim = args.embedding_dim
    epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    dropout = args.dropout

    setup_seed(seed=1023)

    ckpt_path = '{}/{}.ckpt'.format(ckpt_fold, model_name)
    # 检查文件夹是否存在
    if not os.path.exists(ckpt_fold):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(ckpt_fold)
        log.logger.info(f"文件夹'{ckpt_fold}'已创建。")
    else:
        log.logger.info(f"文件夹'{ckpt_fold}'已存在。")

    taobaoData = taobaoDataProcess(log, args.profile_path, args.ad_path, args.user_path, embedding_dim, args.sample_rate)

    # Define Model,train,predict and evaluate
    device = 'cpu'
    if args.use_cuda and torch.cuda.is_available():
        log.logger.info('cuda ready...')
        device = args.cuda_number

    es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=1,
                       patience=2, mode='max', baseline=None)
    mdckpt = ModelCheckpoint(filepath=ckpt_path, monitor='val_auc',
                             mode='max', verbose=1, save_best_only=True, save_weights_only=True)

    # 仅用于非双塔的模型
    linear_feature_columns = taobaoData.user_feature_columns + taobaoData.item_feature_columns
    user_feature_columns = taobaoData.user_feature_columns
    item_feature_columns = taobaoData.item_feature_columns
    dnn_feature_columns = linear_feature_columns

    # model = chooseModel(model_name, user_feature_columns, item_feature_columns, linear_feature_columns, dnn_feature_columns,
    #             dropout, device)

    model = chooseModel(model_name, user_feature_columns, item_feature_columns, linear_feature_columns,
                        dnn_feature_columns, dropout, device, log, data_name="taobao",
                        user_feature_columns_for_recon=taobaoData.user_feature_columns_for_recon,
                        item_feature_columns_for_recon=taobaoData.item_feature_columns_for_recon)
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['auc', 'accuracy', 'logloss'], lr=lr)
    # 因为加了early stopping，所以保留的模型是在验证集上val_auc表现最好的模型
    model.fit(taobaoData.train_model_input, taobaoData.train[taobaoData.target].values, batch_size=batch_size,
              epochs=epoch, verbose=1,
              validation_split=0.2,
              callbacks=[es, mdckpt])

    # 开始模型评估
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # 6.Evaluate
    # 看下最佳模型在完整的训练集上的表现
    eval_tr = model.evaluate(taobaoData.train_model_input, taobaoData.train[taobaoData.target].values)
    log.logger.info(f"model_name = {model_name}; evaluate:{eval_tr}")

    # %%
    pred_ts = model.predict(taobaoData.test_model_input, batch_size=2048)
    log.logger.info(f"model_name = {model_name}; test LogLoss, {round(log_loss(taobaoData.test[taobaoData.target].values, pred_ts), 4)}")
    log.logger.info(f"model_name = {model_name}; test AUC, {round(roc_auc_score(taobaoData.test[taobaoData.target].values, pred_ts), 4)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ["int_tower", "dssm",  "dat", "deep_fm", "dcn", "cold", "auto_int", "wide_and_deep", "tim", "kan_Tim"]
    parser.add_argument("--model_name", type=str, default="tim")
    parser.add_argument("--profile_path", type=str, default="/data/workPlace/recall_model/data/Alibaba/raw_sample.csv")
    parser.add_argument("--ad_path", type=str, default="/data/workPlace/recall_model/data/Alibaba/ad_feature.csv")
    parser.add_argument("--user_path", type=str, default="/data/workPlace/recall_model/data/Alibaba/user_profile.csv")

    # parser.add_argument("--profile_path", type=str, default="./data/Alibaba/raw_sample_test.csv")
    # parser.add_argument("--ad_path", type=str, default="./data/Alibaba/ad_feature.csv")
    # parser.add_argument("--user_path", type=str, default="./data/Alibaba/user_profile.csv")
    parser.add_argument("--ckpt_fold", type=str, default="./checkpoints")
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--cuda_number", type=str, default="cuda:1")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=30)
    # parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--sample_rate", type=float, default=0.3)
    opt = parser.parse_args()
    log = Logger('./log/movielens_models.log', level='debug')
    main(opt, log)

