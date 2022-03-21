# Demo Code for Paper:
# [Title]  - "Video Understanding Based Random Hand Gesture Authentication"
# [Author] - Wenwei Song, Wenxiong Kang, Lu Wang, Zenan Lin, and Mengting Gan
# [Github] - https://github.com/SCUT-BIP-Lab/3DTDS-Net

import torch
from model.tds_net_3d import Model_3DTDS_Net
from dataset.frame_dataloader import FrameDataloader
# from loss.loss import AMSoftmax
# from dataset.transforms import resnet_train_transform, resnet_eval_transform


def feedforward_demo(frame_length, feature_dim, out_dim):
    model = Model_3DTDS_Net(frame_length=frame_length, feature_dim=feature_dim, out_dim=out_dim)
    # AMSoftmax loss function
    # criterian = AMSoftmax(in_feats=self.feature_dim, n_classes=143)
    # there are 143 identities in the training set
    data = torch.randn(2, 20, 3, 224, 224) #batch, frame, channel, h, w
    data = data.view(-1, 3, 224, 224) #regard the frame as batch
    physical_feature, behavior_feature = model(data) # feedforward
    # global temporal avg pool
    physical_feature = torch.mean(physical_feature, dim=1, keepdim=False)
    # BM-Fusion
    physical_feature = torch.div(physical_feature,
                                 torch.norm(physical_feature, p=2, dim=1, keepdim=True).clamp(min=1e-12))
    cv_feature = 0.5 * physical_feature + 0.5 * behavior_feature
    cv_feature = torch.div(cv_feature, torch.norm(cv_feature, p=2, dim=1, keepdim=True).clamp(min=1e-12))
    # then use the cv_feature to calculate the EER when testing or to calculate the loss when training
    # loss, costh = self.criterian(cv_feature, label) # when training
    return cv_feature


def random_hand_gesture_load_demo(frame_len, transform, is_train):
    video_path = "/data/DHGA/" # dataset root
    loader = FrameDataloader(frame_len=frame_len, transform=transform, is_train=is_train)
    random_hand_gesture = loader.getVideoFrameWithTransform(video_path)
    return random_hand_gesture


if __name__ == '__main__':
    # there are 20 frames in each random hand gesture video
    frame_length = 20
    # the feature dim of last feature map (layer4) from ResNet18 is 512
    feature_dim = 512
    # the identity feature dim
    out_dim = 512
    # feedforward process
    feedforward_demo(frame_length, feature_dim, out_dim)
    # random hand gesture loading process
    random_hand_gesture_load_demo(frame_length, None, is_train=True)  # for demo
    # random_hand_gesture_load_demo(frame_length, resnet_train_transform, is_train=True) # for training
    # random_hand_gesture_load_demo(frame_length, resnet_train_transform, is_train=True)  # for testing
    print("Demo is finished!")