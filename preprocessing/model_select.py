
from model.IntTower import IntTower
from model.dssm import DSSM
from model.deepfm import DeepFM
from model.dcn import DCN
from model.dat import DAT
from model.cold import Cold
from model.autoint import AutoInt
from model.wdm import WideDeep
from model.hit import HitTower
from model.poly_encoder import PolyEncoder
from model.mvke import MVKE
from model.KAN_TimTower import KanTimTower


def chooseModel(model_name, user_feature_columns, item_feature_columns, linear_feature_columns,
                dnn_feature_columns, dropout, device, log, data_name=None,
                user_feature_columns_for_recon=None, item_feature_columns_for_recon=None,
                ouput_head=2):
    if model_name == "int_tower":
        log.logger.info("model_name int_tower")
        model = IntTower(user_feature_columns, item_feature_columns, field_dim=16, task='binary', dnn_dropout=dropout,
                         device=device, user_head=32, item_head=32, user_filed_size=9, item_filed_size=6)
    elif model_name == "dssm":
        log.logger.info("model_name dssm")
        model = DSSM(user_feature_columns, item_feature_columns, task='binary', device=device)
    elif model_name == "poly_encoder":
        log.logger.info("model_name poly_encoder")
        model = PolyEncoder(user_feature_columns, item_feature_columns, dnn_dropout=dropout,
                            task='binary', device=device)
    elif model_name == "MVKE":
        log.logger.info("model_name MVKE")
        model = MVKE(user_feature_columns, item_feature_columns, dnn_dropout=dropout, task='binary', device=device)
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
    elif model_name == "hit":
        log.logger.info("model_name hit")
        field_dim = 16
        hidden_dim = ouput_head * field_dim
        model = HitTower(user_feature_columns, item_feature_columns,
                         user_input_for_recon=user_feature_columns_for_recon,
                         item_input_for_recon=item_feature_columns_for_recon, field_dim = field_dim,
                         user_head=ouput_head, item_head=ouput_head,
                         task='binary', dnn_dropout=dropout, device=device, activation_for_recon='relu',
                         hidden_units_for_recon=(128, hidden_dim), use_target=True, use_non_target=True)
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
