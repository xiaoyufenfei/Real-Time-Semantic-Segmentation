import os
def get_model(model_name,classes,pre_train=False,mode = 'train'):
    if model_name == 'ESpnet_2_8_decoder':
        from models import Espnet
        if pre_train:
            pre_train_path = os.path.join(pre_train)
            model = Espnet.ESPNet(classes, 2, 8, pre_train_path,mode=mode)
        else:
            model = Espnet.ESPNet(classes, 2, 8,mode=mode)
    elif model_name == 'ESpnet_2_8':
        from models import Espnet
        model = Espnet.ESPNet_Encoder(classes, 2, 8)
    elif model_name == 'EDAnet':
        from models import EDANet
        model = EDANet.EDANet(classes)
    elif model_name == 'ERFnet':
        from models import ERFnet
        model = ERFnet.Net(classes)
    elif model_name == 'Enet':
        from models import Enet
        model = Enet.ENet(classes)
    elif model_name == 'IRRnet_2_8':
        from models import Irregularity_conv
        model = Irregularity_conv.ESPNet(classes, 2, 8, mode=mode)
    elif model_name == 'MOBILE_V2':
        from models import mobilenet
        model = mobilenet.mbv2(classes)
    elif model_name == 'RF_LW_resnet_50':
        from models import resnet
        model = resnet.rf_lw50(classes)
    elif model_name == 'RF_LW_resnet_101':
        from models import resnet
        model = resnet.rf_lw101(classes)
    elif model_name == 'RF_LW_resnet_152':
        from models import resnet
        model = resnet.rf_lw152(classes)
    elif model_name == 'Bisenet':
        from models import BiSeNet
        model = BiSeNet.BiSeNet(out_class=classes)
    elif model_name == 'Basenet':
        from models import Basenet
        model = Basenet.Basenet(classes)
    else:
        raise NotImplementedError
    return model