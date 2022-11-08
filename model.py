import segmentation_models_pytorch as smp
import torch
from torch import nn
import AttentionUnet
from Unet import Unet
import RAT_Unet
from rga_resnet import resnet50_rga
from torchinfo import summary
import Unet_Segmentation_Pytorch_Nest_of_Unets.Models as spm
from torch.backends import cudnn
import Unet_Segmentation_Pytorch_Nest_of_Unets.AtteffUnet as atteffunet
import SwinUnet
import Unet_3Plus


def build_model(CFG):
    if (CFG['model_name'] == 'Unet'):
        model = smp.Unet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        decoder_use_DIA = False,
                    ).to(CFG['device'])
        print('model is Unet')
    elif (CFG['model_name'] == 'SCSE_Unet'):
        model = smp.Unet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        decoder_attention_type=CFG['attention'],
                        decoder_use_DIA=False,
                    ).to(CFG['device'])
        # print('model is vgg16-SCSE_Unet')
        print('model is efficientnet-b0-SCSE_Unet')
    elif (CFG['model_name'] == 'UnetPlusPlus'):
        model = smp.UnetPlusPlus(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                        decoder_use_DIA = False,
                    ).to(CFG['device'])
        print('model is UnetPlusPlus')
    elif (CFG['model_name'] == 'Unet_3Plus'):
        model = torch.nn.DataParallel(Unet_3Plus.UNetPPP(
                        in_channels=3,
                        num_classes=3,
                    )).cuda()
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[CFG['local_rank']], output_device=CFG['local_rank'])
        print('model is Unet_3Plus')
    elif (CFG['model_name'] == 'DeepLabV3'):
        model = smp.DeepLabV3(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                    ).to(CFG['device'])
        print('model is DeepLabV3')
    elif (CFG['model_name'] == 'FPN'):
        model = smp.FPN(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                    ).to(CFG['device'])
        print('model is FPN')
    elif (CFG['model_name'] == 'SwinUnet'):
        model = SwinUnet.SwinUnet(
                    ).to(CFG['device'])
        print('model is SwinUnet')
    elif (CFG['model_name'] == 'MAnet'):
        model = smp.MAnet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                    ).to(CFG['device'])
        print('model is MAnet')
    elif (CFG['model_name'] == 'Linknet'):
        model = smp.Linknet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                    ).to(CFG['device'])
        print('model is Linknet')
    elif (CFG['model_name'] == 'AttentionUnet'):
        if (CFG['backbone'] == 'efficientnet-b0'):
            model = atteffunet.get_efficientunet_b0(out_channels=3, concat_input=True, pretrained=False).to(CFG['device'])
            print('model is efficientnet-b0-AttentionUnet')
        else:
            model = spm.AttU_Net(img_ch=3, output_ch=3).to(CFG['device'])
            print('model is vgg16-AttentionUnet')
    elif (CFG['model_name'] == 'SIAUnet'):
        model = smp.Unet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                        decoder_use_DIA = CFG['use_channel_attention'],
                        decoder_use_DBM = CFG['use_DBM'],
                        decoder_use_timestamp = CFG['use_timestamp'],
                        batch_size = CFG['train_bs']
                    ).to(CFG['device'])
        print('model is vgg16-DIAUnet')
    elif (CFG['model_name'] == 'TimesUnet'):
        model = smp.Unet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                        decoder_use_DIA=CFG['use_channel_attention'],
                        decoder_use_DBM=CFG['use_DBM'],
                        decoder_use_timestamp=CFG['use_timestamp'],
                        batch_size=CFG['train_bs']
        ).to(CFG['device'])
        print('model is TimesUnet')
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

