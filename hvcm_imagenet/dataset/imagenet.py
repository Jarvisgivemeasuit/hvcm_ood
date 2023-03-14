import os
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

cv2.setNumThreads(1)

class Imagenet(Dataset):
    '''
    Generating a torch dataset using Imagenet for training or validation.
    Args:
        mode: Specifies the dataset to train or test.
        data_domain: Determines this dataset as IND or OOD.
    '''
    def __init__(self, mode, data_path, num_cls, transform=None) -> None:
        super().__init__()
        assert mode in ['train', 'val']

        self.mode = mode
        self.transform = transform
        self.imagenet_path = os.path.join(data_path, mode)
        self.classes, self.img_list = {}, []

        with open(f'dataset/ind_imagenet_{num_cls}cls.txt', 'r') as f:
            for idx, line in enumerate(f):

                cls_name = line.strip()
                self.classes[cls_name] = idx

                cls_img_list = os.listdir(os.path.join(self.imagenet_path, cls_name))
                cls_img_list = [os.path.join(cls_name, k) for k in cls_img_list]
                self.img_list = self.img_list + cls_img_list

    def __getitem__(self, idx):

        img_name = self.img_list[idx]

        cls_name = img_name.split('/')[0]
        cls_label = self.classes[cls_name]

        img = cv2.imread(os.path.join(self.imagenet_path, img_name), cv2.IMREAD_COLOR)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        else:
            img = self.train_transforms(img) if self.mode == 'train' else \
                self.val_transforms(img)

        return img, cls_label

    def __len__(self):
        return len(self.img_list)

    def train_transforms(self, image):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(224, (0.6, 1)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return transform(image)

    def val_transforms(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return transform(image)
