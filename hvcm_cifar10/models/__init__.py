from .resnet import *
from .resnet_gram import *

def load_model(name, num_classes=10, pretrained=False, **kwargs):
    model_dict = globals()
    model = model_dict[name](pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model
