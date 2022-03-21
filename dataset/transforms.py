# pip install git+https://github.com/hassony2/torch_videovision
from torchvideotransforms import video_transforms, volume_transforms

resnet_train_transform = video_transforms.Compose([
        video_transforms.RandomRotation(15),
        video_transforms.Resize((224, 224)),
        video_transforms.ColorJitter(0.3, 0.3, 0.3),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

resnet_eval_transform = video_transforms.Compose([
        video_transforms.Resize((224, 224)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])