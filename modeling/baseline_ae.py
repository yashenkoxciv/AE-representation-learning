import torch.nn as nn
import torchvision.models as models

from .generator import Generator


def backbone(model, imagenet_pretrained=False):
    m = getattr(models, model)(pretrained=imagenet_pretrained)
    return m


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class ResNet18AE(nn.Module):
    def __init__(self, z_dim, pretrained_backbone, fine_tuning):
        super(ResNet18AE, self).__init__()
        # create encoder network
        self.backbone = backbone('resnet18', pretrained_backbone)
        # replace last layer with gag
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential()
        # disable backbone learning if we are going to fine tune
        set_parameter_requires_grad(self.backbone, fine_tuning)
        # add representation layer
        self.fc = nn.Linear(in_features, z_dim)
        # let's create generator network
        self.generator = Generator(None, image_size=64, z_dim=z_dim, conv_dim=64)

    def forward(self, x):
        f = self.backbone(x)
        z = self.fc(f)
        rx = self.generator(z)
        return rx

    def encode(self, x):
        f = self.backbone(x)
        z = self.fc(f)
        # add normalization
        return z

    def decode(self, z):
        gx = self.generator(z)
        return gx
