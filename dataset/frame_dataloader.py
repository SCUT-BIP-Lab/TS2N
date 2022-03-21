# Rand Hand Generation Code for Paper:
# [Title]  - "Video Understanding Based Random Hand Gesture Authentication"
# [Author] - Wenwei Song, Wenxiong Kang, Lu Wang, Zenan Lin, and Mengting Gan
# [Github] - https://github.com/SCUT-BIP-Lab/3DTDS-Net

import os
import random
import numpy as np
from PIL import Image
from numpy.random import randint

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

class FrameDataloader(object):
    def __init__(self, frame_len, transform, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.frame_len = frame_len
        self.extensions = ".jpg"
        self.transform = transform

    def getVideoFrameWithTransform(self, video_path):
        try:
            # filelist = os.listdir(video_path)
            # filelist = [n for n in filelist if has_file_allowed_extension(n, self.extensions)]
            # filelist.sort()
            filelist = [i for i in range(64)] #just for demo, 64 is the video frame num

            video_frame_len = len(filelist)

            if self.is_train:
                blockStartPos = random.randint(0, video_frame_len - self.frame_len)
            else:
                blockStartPos = (self.frame_len + 1) // 2 - 10 - 1  # -1 to zero

            slicelist = [i for i in range(blockStartPos, blockStartPos + self.frame_len)] #the random hand gesture frames
            print("the random hand gesture frame ID is:")
            print(slicelist)

            imglist = []

            # for idx in slicelist:
            #     img = Image.open(os.path.join(video_path, filelist[idx]))
            #     img = img.convert("RGB")
            #     imglist.append(img)
            # imglist = self.transform(imglist)  # type torch, shape(3, k, 224, 224), k是帧数
            return imglist

        except Exception as e:
            print(e)
            print(video_path)
