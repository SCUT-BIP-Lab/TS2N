# Demo Code for Paper:
# [Title]  - "Random Hand Gesture Authentication via Efficient Temporal Segment Set Network"
# [Author] - Yihong Lin, Wenwei Song, Wenxiong Kang

import torch
from model.ts2n import Model_Resnet_Att, Model_Gaitset_PAN


def feedforward_demo(frame_length, feature_dim, out_dim):
    physical_model = Model_Resnet_Att(frames_per_video=frame_length, feature_dim=feature_dim, out_dim=out_dim)
    behavior_model = Model_Gaitset_PAN(frames_per_video=frame_length, feature_dim=feature_dim, out_dim=out_dim)

    # there are 143 identities in the training set
    data = torch.randn(2, 20, 3, 224, 224) #batch, frame, channel, h, w
    physical_feature, behavior_feature = physical_model(data), behavior_model(data) # feedforward
    
    return {physical_feature, behavior_feature}


if __name__ == '__main__':
    # there are 20 frames in each random hand gesture video
    frame_length = 20
    # the feature dim of last feature map (layer4) from ResNet18 is 512
    feature_dim = 512
    # the identity feature dim
    out_dim = 512
    # feedforward process
    feedforward_demo(frame_length, feature_dim, out_dim)
    print("Demo is finished!")