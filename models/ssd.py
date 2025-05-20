from torchvision.models.detection.ssd import SSD300_VGG16_Weights, SSDClassificationHead

from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

def get_ssd(num_classes=3, pretrained=True):
    if pretrained:
        weights = SSD300_VGG16_Weights.DEFAULT
    else:
        weights = None
    model = ssd300_vgg16(weights=weights, num_classes=num_classes)
    return model