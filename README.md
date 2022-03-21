# 3DTDS-Net: 3D Temporal Difference Symbiotic Neural Network 
Pytorch Implementation of paper:

> **Video Understanding Based Random Hand Gesture Authentication**
>
> Wenwei Song, Wenxiong Kang\*, Lu Wang, Zenan Lin and Mengting Gan.

## Main Contribution
Existing hand gesture authentication methods require the probe gesture types to be consistent with
 the registered ones, which reduces the user-friendliness and efficiency of authentication. 
 In this paper, a video understanding based random hand gesture authentication method is introduced to eliminate 
 this limitation, in which users only need to improvise a random hand gesture in front of an RGB camera without memory 
 and hesitation in both the enrollment and verification stage. The random hand gesture is a promising biometric trait 
 containing both physiological and behavioral characteristics. To fully unleash the potential of random hand gesture 
 authentication, we design a simple but effective behavior representation (modality), temporal difference map, for better 
 behavioral characteristic understanding and present an efficient model called 3D Temporal Difference Symbiotic Neural 
 Network (3DTDS-Net) that can separately extract physiological and behavioral features as well as automatically assign 
 fusion weights for the two features to complement each other’s strengths based on the magnitude of behavioral features 
 in an end-to-end fashion. We also adapt and reimplement 17 SOTA neural networks for authentication from other tasks, 
 such as action classification and gait recognition, to make convincing comparisons. Extensive experiments on the 
 SCUT-DHGA dataset demonstrate the effectiveness of temporal difference maps and the superiority of 3DTDS-Net. 

<p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/3DTDS-Net/main/img/3DTDS-Net.png" />
</p>

## Comparisons with SOTAs
Our 3DTDS-Net achieves very competitive performance while enjoying low computation costs for 
fast random hand gesture authentication on SCUT-DHGA dataset.
<p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/3DTDS-Net/main/img/SOTA_Comparison.png" />
</p>


## Dependencies
Please make sure the following libraries are installed successfully:
- [PyTorch](https://pytorch.org/) >= 1.7.0

## How to use
This repository is a demo of 3DTDS-Net. Through debugging ([main.py](/main.py)), you can quickly understand the 
configuration and building method ([tds_net_3d](/model/tds_net_3d.py)) of 3DTDS-Net, as well as
the random hand gesture loading strategy ([frame_dataloader](/dataset/frame_dataloader.py)).

If you want to explore the entire dynamic hand gesture authentication framework, please refer to our pervious work [SCUT-DHGA](https://github.com/SCUT-BIP-Lab/SCUT-DHGA) 
or send an email to Prof. Kang ([email](auwxkang@scut.edu.cn)).


