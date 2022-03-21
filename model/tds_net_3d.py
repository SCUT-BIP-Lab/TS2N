# 3DTDS-Net Model Code for Paper:
# [Title]  - "Video Understanding Based Random Hand Gesture Authentication"
# [Author] - Wenwei Song, Wenxiong Kang, Lu Wang, Zenan Lin, and Mengting Gan
# [Github] - https://github.com/SCUT-BIP-Lab/3DTDS-Net

import torch
import torch.nn as nn
import torchvision


class Model_3DTDS_Net(torch.nn.Module):
    def __init__(self, frame_length, feature_dim, out_dim):
        super(Model_3DTDS_Net, self).__init__()
        # load the pretrained ResNet18
        self.model = torchvision.models.resnet18(pretrained=True)
        # change the last fc with the shape of 512×143
        self.model.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)

        # the feature dim of last feature map (layer4) from ResNet18 is 512
        self.feature_dim = feature_dim
        self.out_dim = out_dim # the identity feature dim

        # there are 20 frames in each random hand gesture video
        self.frame_length = frame_length
        temporal_difference_num = self.frame_length - 1

        # build the five convolutions in the symbiotic branch
        self.behavior_layer0 = self._make_symbiotic_layer_3d(1, 2, kernel_size=(3, 7, 7), padding=(1, 3, 3), pool=True)
        self.behavior_layer1 = self._make_symbiotic_layer_3d(2, 4, stride=(1, 1, 1))
        self.behavior_layer2 = self._make_symbiotic_layer_3d(3, 6)
        self.behavior_layer3 = self._make_symbiotic_layer_3d(4, 8)
        self.behavior_layer4 = self._make_symbiotic_layer_3d(5, 10)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # there are 5 channel concatenation ops in the symbiotic branch
        self.behavior_fc = nn.Linear(in_features=temporal_difference_num * 5, out_features=512)

    def _make_symbiotic_layer_3d(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), pool=False):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConstantPad3d(padding=(0, 0, 0, 0, 1, 0), value=0),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)) # for channel interlacing
        )
        if pool:
            layer.add_module("pooling", nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        return layer

    def getBehaviorFeature(self, physical_feature): #interframe subtraction-pointwise summation
        physical_feature = physical_feature.view((-1, self.frame_length) + physical_feature.shape[-3:])
        behavior_feature = physical_feature[:, :self.frame_length - 1, :, :, :] - physical_feature[:, 1:self.frame_length, :, :, :]
        behavior_feature = torch.sum(behavior_feature, 2)
        return behavior_feature

    def channel_interlace(self, feature):
        batch, channel, frame, w, h = feature.shape
        feature = feature.view(batch, channel // 2, 2, frame, w, h)
        feature = feature.permute(0, 1, 3, 2, 4, 5).contiguous()
        feature = feature.view(batch, channel // 2, frame * 2, w, h)[:, :, :-1]
        return feature

    def behavior_block_forward(self, layer, physical_feature, behavior_feature, is_cat=True):
        if behavior_feature is None:
            behavior_feature = self.getBehaviorFeature(physical_feature).unsqueeze(1)
        else:
            # channel concatenation in the symbiotic branch
            if is_cat:
                behavior_feature = torch.cat((behavior_feature, self.getBehaviorFeature(physical_feature).unsqueeze(1)), dim=1)
        # get the symbiotic layer
        behavior_func = "behavior_layer" + str(layer)
        behavior_feature = getattr(self, behavior_func)(behavior_feature)
        behavior_feature = self.channel_interlace(behavior_feature)
        return behavior_feature

    def forward(self, data, label=None):
        #get the temporal difference map from raw RGB
        behavior_feature = self.behavior_block_forward(layer=0, physical_feature=data, behavior_feature=None,
                                                       is_cat=False)
        physical_feature = self.model.conv1(data)
        physical_feature = self.model.bn1(physical_feature)
        physical_feature = self.model.relu(physical_feature)
        physical_feature = self.model.maxpool(physical_feature)

        for i in range(1, 4): #layer1, layer2, layer3
            behavior_feature = self.behavior_block_forward(layer=i, physical_feature=physical_feature,
                                                           behavior_feature=behavior_feature, is_cat=True)
            resnet_layer = getattr(self.model, "layer"+str(i))
            physical_feature = resnet_layer(physical_feature)

        behavior_feature = self.behavior_block_forward(layer=4, physical_feature=physical_feature,
                                                       behavior_feature=behavior_feature, is_cat=True)

        physical_feature = self.model.layer4(physical_feature)
        physical_feature = self.model.avgpool(physical_feature)
        physical_feature = torch.flatten(physical_feature, 1)
        physical_feature = self.model.fc(physical_feature)
        physical_feature = physical_feature.view(-1, self.frame_length, self.feature_dim)

        behavior_feature = self.avgpool(behavior_feature)
        behavior_feature = torch.flatten(behavior_feature, 1)
        behavior_feature = self.behavior_fc(behavior_feature)

        return physical_feature, behavior_feature

