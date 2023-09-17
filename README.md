# Random Hand Gesture Authentication via Efficient Temporal Segment Set Network 
Pytorch Implementation of paper:

> **Random Hand Gesture Authentication via Efficient Temporal Segment Set Network**
>
> Yihong Lin, Wenwei Song and Wenxiong Kang\*.

## Main Contribution
Biometric authentication technologies are rapidly gaining popularity, and hand gestures are emerging as a promising biometric trait due to their rich physiological and behavioral characteristics. Hand gesture authentication can be categorized as defined hand gesture authentication and random hand gesture authentication. Unlike defined hand gesture authentication, random hand gesture authentication is not constrained to specific hand gesture types, allowing users to perform hand gestures randomly during enrollment and verification, thus more flexible and friendly. However, in random hand gesture authentication, the model needs to extract more generalized physiological and behavioral features from different viewpoints and positions without gesture templates, which is more challenging. In this paper, we present a novel efficient Temporal-Segment-Set-Network (TS2N) that directly extracts both behavioral and physiological features from a single RGB video to further enhance the security of random hand gesture authentication. Our method adopts a motion pseudo-modality and leverages a set-based representation to capture behavioral characteristics online. Additionally, we propose a new channel-spatial attention mechanism, Contextual Squeeze-and-Excitation Network (CoSEN), to better abstract and understand physiological characteristics by explicitly modeling the channel and spatial interdependence, thereby adaptively recalibrating channel-specific and spatial-specific responses. Extensive experiments on the largest public hand gesture authentication dataset SCUT-DHGA demonstrate TS2N's superiority against 21 state-of-the-art models in terms of EER (5.707\% for full version and 6.664\% for lite version) and computational cost (98.9022G for full version and 46.3741G for lite version), showing a promising avenue for secure and efficient biometric authentication systems.

<!-- <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/3DTDS-Net/main/img/3DTDS-Net.png" />
</p> -->
<p align="center">
  <img src="/img/TS2N.png" />
</p>

## Dependencies
Please make sure the following libraries are installed successfully:
- [PyTorch](https://pytorch.org/) >= 1.7.0

## How to use
This repository is a demo of TS2N. Through debugging ([main.py](/main.py)), you can quickly understand the 
configuration and building method ([ts2n](/model/ts2n.py)) of TS2N.

If you want to explore the entire dynamic hand gesture authentication framework, please refer to our pervious work [SCUT-DHGA](https://github.com/SCUT-BIP-Lab/SCUT-DHGA) 
or send an email to Prof. Kang (auwxkang@scut.edu.cn).


