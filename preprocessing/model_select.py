
from model.IntTower import IntTower
from model.dssm import DSSM
from model.deepfm import DeepFM
from model.dcn import DCN
from model.dat import DAT
from model.cold import Cold
from model.autoint import AutoInt
from model.wdm import WideDeep
from model.tim import TimTower
from model.KAN_TimTower import KanTimTower


def chooseModel(model_name, user_feature_columns, item_feature_columns, linear_feature_columns,
                dnn_feature_columns, dropout, device, log, data_name=None,
                user_feature_columns_for_recon=None, item_feature_columns_for_recon=None):
    if model_name == "int_tower":
        log.logger.info("model_name int_tower")
        if data_name == "Amazon":
            model = IntTower(user_feature_columns, item_feature_columns, field_dim=64, task='binary',
                             dnn_dropout=dropout, device=device, user_head=32, item_head=32, user_filed_size=1,
                             item_filed_size=2)
        else:
            model = IntTower(user_feature_columns, item_feature_columns, field_dim=64, task='binary',
                             dnn_dropout=dropout, device=device, user_head=2, item_head=2, user_filed_size=5,
                             item_filed_size=2)
    elif model_name == "dssm":
        log.logger.info("model_name dssm")
        model = DSSM(user_feature_columns, item_feature_columns, task='binary', device=device)
    elif model_name == "dat":
        log.logger.info("model_name dat")
        model = DAT(user_feature_columns, item_feature_columns, task='binary', dnn_dropout=dropout,
                    device=device)
    elif model_name == "deep_fm":
        log.logger.info("model name deep_fm")
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout,
                       device=device)
    elif model_name == "dcn":
        log.logger.info("model_name dcn")
        model = DCN(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout,
                    device=device)
    elif model_name == "cold":
        log.logger.info("model_name cold")
        model = Cold(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout,
                     device=device)
    elif model_name == "auto_int":
        log.logger.info("model_name auto_int")
        model = AutoInt(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout,
                        device=device)
    elif model_name == "wide_and_deep":
        log.logger.info("model_name wide_and_deep")
        model = WideDeep(linear_feature_columns, dnn_feature_columns, task='binary',
                         device=device)
    elif model_name == "tim":
        log.logger.info("model_name tim")
        if data_name == "movieLens":
            model = TimTower(user_feature_columns, item_feature_columns,
                             user_input_for_recon=user_feature_columns_for_recon,
                             item_input_for_recon=item_feature_columns_for_recon, field_dim = 16,
                             task='binary', dnn_dropout=dropout, device=device, activation_for_recon='relu',
                             hidden_units_for_recon=(32, 32), use_target=True, use_non_target=True)
        else:
            model = TimTower(user_feature_columns, item_feature_columns,
                             user_input_for_recon=user_feature_columns_for_recon,
                             item_input_for_recon=item_feature_columns_for_recon, field_dim = 8,
                             task='binary', dnn_dropout=dropout, device=device, activation_for_recon='relu',
                             hidden_units_for_recon=(32, 16), use_target=True, use_non_target=True)
    elif model_name == "kanTim":
        log.logger.info("model_name kanTim")
        model = KanTimTower(user_feature_columns, item_feature_columns,
                         user_input_for_recon=user_feature_columns_for_recon,
                         item_input_for_recon=item_feature_columns_for_recon,
                         task='binary', dnn_dropout=dropout, device=device, activation_for_recon='relu',
                         use_target=True, use_non_target=True)
    else:
        log.logger.info("model_name wide_and_deep")
        model = WideDeep(linear_feature_columns, dnn_feature_columns, task='binary',
                         device=device)
        raise ValueError("There is no such value for model_name")

    return model