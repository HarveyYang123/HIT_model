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
from preprocessing.dataProcess import movieDataProcess, setup_seed

def main(args, log):
    use_cuda, cuda_number = True, 'cuda:0'
    # ["int_tower", "dssm",  "dat", "deep_fm", "dcn", "cold", "auto_int", "wide_and_deep", "tim_tower"]
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
        log.logger.info(f"the fold '{ckpt_fold}' is created now.")
    else:
        log.logger.info(f"the fold '{ckpt_fold}' already exists.")

    movieData = movieDataProcess(log, args.data_path, embedding_dim)

    # Define Model,train,predict and evaluate
    device = 'cpu'
    if use_cuda and torch.cuda.is_available():
        log.logger.info('cuda ready...')
        device = cuda_number


    # es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=1,
    #                    patience=5, mode='max', baseline=None)
    # mdckpt = ModelCheckpoint(filepath=ckpt_path, monitor='val_auc',
    #                          mode='max', verbose=1, save_best_only=True, save_weights_only=True)
    es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=1,
                       patience=2, mode='max')
    mdckpt = ModelCheckpoint(filepath=ckpt_path, monitor='val_auc',
                             mode='max', verbose=1, save_best_only=True, save_weights_only=True)

    # 仅用于非双塔的模型
    linear_feature_columns = movieData.user_feature_columns + movieData.item_feature_columns
    user_feature_columns = movieData.user_feature_columns
    item_feature_columns = movieData.item_feature_columns
    dnn_feature_columns = linear_feature_columns

    model = chooseModel(model_name, user_feature_columns, item_feature_columns, linear_feature_columns,
                        dnn_feature_columns, dropout, device, log, data_name="movieLens",
                        user_feature_columns_for_recon=movieData.user_feature_columns_for_recon,
                        item_feature_columns_for_recon=movieData.item_feature_columns_for_recon)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['auc', 'accuracy', 'logloss']
                  , lr=lr)
    # 因为加了early stopping，所以保留的模型是在验证集上val_auc表现最好的模型
    model.fit(movieData.train_model_input, movieData.train[movieData.target].values, batch_size=batch_size, epochs=epoch, verbose=2,
              validation_split=0.2,
              callbacks=[es, mdckpt])

    # 开始模型评估
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()


    # %%
    # 6.Evaluate
    # 看下最佳模型在完整的训练集上的表现
    eval_tr = model.evaluate(movieData.train_model_input, movieData.train[movieData.target].values)
    log.logger.info(f"model_name = {model_name}; evaluate:{eval_tr}")

    # %%
    pred_ts = model.predict(movieData.test_model_input, batch_size=2048)
    log.logger.info(f"model_name = {model_name}; test LogLoss, {round(log_loss(movieData.test[movieData.target].values, pred_ts), 4)}")
    log.logger.info(f"model_name = {model_name}; test AUC, {round(roc_auc_score(movieData.test[movieData.target].values, pred_ts), 4)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ["int_tower", "dssm",  "dat", "deep_fm", "dcn", "cold", "auto_int", "wide_and_deep", "tim", "kanTim"]
    parser.add_argument("--model_name", type=str, default="int_tower")
    parser.add_argument("--data_path", type=str, default="./data/movielens.txt")
    # parser.add_argument("--data_path", type=str, default="./data/movielens_test.txt")
    parser.add_argument("--ckpt_fold", type=str, default="./checkpoints")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--random_seed", type=int, default=1023)
    opt = parser.parse_args()
    log = Logger('./log/movielens_models.log', level='debug')
    main(opt, log)
    # branch_test
    # branch_test123
    # test_updata

