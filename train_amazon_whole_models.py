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
from preprocessing.dataProcess import amazonDataProcess, setup_seed

def main(args, log):
    model_name = args.model_name
    ckpt_fold = args.ckpt_fold

    embedding_dim = args.embedding_dim
    epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    dropout = args.dropout

    setup_seed(seed=args.random_seed)

    ckpt_path = '{}/{}.ckpt'.format(ckpt_fold, model_name)
    # 检查文件夹是否存在
    if not os.path.exists(ckpt_fold):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(ckpt_fold)
        log.logger.info(f"文件夹'{ckpt_fold}'已创建。")
    else:
        log.logger.info(f"文件夹'{ckpt_fold}'已存在。")

    AmazonData = amazonDataProcess(log, args.data_path, embedding_dim)

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
    linear_feature_columns = AmazonData.user_feature_columns + AmazonData.item_feature_columns
    user_feature_columns = AmazonData.user_feature_columns
    item_feature_columns = AmazonData.item_feature_columns
    dnn_feature_columns = linear_feature_columns
    
    model = chooseModel(model_name, user_feature_columns, item_feature_columns, linear_feature_columns,
                        dnn_feature_columns, dropout, device, log, data_name="Amazon",
                        user_feature_columns_for_recon=AmazonData.user_feature_columns_for_recon,
                        item_feature_columns_for_recon=AmazonData.item_feature_columns_for_recon)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['auc', 'accuracy', 'logloss'], lr=lr)
    # 因为加了early stopping，所以保留的模型是在验证集上val_auc表现最好的模型
    model.fit(AmazonData.train_model_input, AmazonData.train[AmazonData.target].values, batch_size=batch_size, epochs=epoch, verbose=2,
              validation_split=0.2,
              callbacks=[es, mdckpt])

    # 开始模型评估
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # Evaluate
    # 看下最佳模型在完整的训练集上的表现
    eval_tr = model.evaluate(AmazonData.train_model_input, AmazonData.train[AmazonData.target].values)
    log.logger.info(f"model_name = {model_name}; evaluate:{eval_tr}")

    # %%
    pred_ts = model.predict(AmazonData.test_model_input, batch_size=128)
    log.logger.info(f"model_name = {model_name}; test LogLoss, {round(log_loss(AmazonData.test[AmazonData.target].values, pred_ts), 4)}")
    log.logger.info(f"model_name = {model_name}; test AUC, {round(roc_auc_score(AmazonData.test[AmazonData.target].values, pred_ts), 4)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ["int_tower", "dssm",  "dat", "deep_fm", "dcn", "cold", "auto_int", "wide_and_deep", "hit"]
    parser.add_argument("--model_name", type=str, default="hit")
    parser.add_argument("--data_path", type=str, default="./data/amazon_eletronics.csv")
    parser.add_argument("--ckpt_fold", type=str, default="./checkpoints/amazon")
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--cuda_number", type=str, default="cuda:1")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=30)
    # parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--random_seed", type=int, default=1023)
    opt = parser.parse_args()
    log = Logger('./log/amazon_data.log', level='debug')
    main(opt, log)

