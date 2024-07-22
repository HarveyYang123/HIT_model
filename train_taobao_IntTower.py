import numpy as np
import pandas as pd
import torch
import torchvision
import random
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import log_loss, roc_auc_score

from model.IntTower import IntTower
from preprocessing.logging import Logger
from preprocessing.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing.dataProcess import taobaoDataProcess, setup_seed


if __name__ == "__main__":
    embedding_dim = 32
    epoch = 15
    batch_size = 2048
    dropout = 0.5
    seed = 1023
    lr = 0.0001

    print("1")

    setup_seed(seed)

    profile_path = '/data/workPlace/recall_model/data/Alibaba/raw_sample.csv'
    ad_path = '/data/workPlace/recall_model/data/Alibaba/ad_feature.csv'
    user_path = '/data/workPlace/recall_model/data/Alibaba/user_profile.csv'
    log = Logger('./log/movielens_models.log', level='debug')
    data = taobaoDataProcess(log, profile_path, ad_path, user_path, embedding_dim)

    # %%
    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:1'

    es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=1,
                       patience=3, mode='max', baseline=None)
    mdckpt = ModelCheckpoint(filepath='fe_model_2.ckpt', monitor='val_auc',
                             mode='max', verbose=1, save_best_only=True, save_weights_only=True)
    model = IntTower(data.user_feature_columns, data.item_feature_columns, field_dim= 16, task='binary', dnn_dropout=dropout,
           device=device, user_head=4,item_head=4,user_filed_size=9,item_filed_size=6)

    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy', 'logloss']
                  , lr=lr)

    model.fit(data.train_model_input, data.train[data.target].values, batch_size=batch_size,
              epochs=epoch, verbose=2, validation_split=0.2, callbacks=[es, mdckpt])

    # 5.preprocess the test data
    model.load_state_dict(torch.load('fe_model_2.ckpt'))
    # 测试时不启用 BatchNormalization 和 Dropout
    model.eval()

    pred_ts = model.predict(data.test_model_input, batch_size=500)

    print("test LogLoss", round(log_loss(data.test[data.target].values, pred_ts), 4))
    print("test AUC", round(roc_auc_score(data.test[data.target].values, pred_ts), 4))


