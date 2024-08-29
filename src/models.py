from torchvision import models
import torch.nn as nn


class FruitClassifier(nn.Module):
    def __init__(self, num_fruit_types, num_colors, resnet_version="resnet50"):
        super(FruitClassifier, self).__init__()
        # Load pretrained resnet50 model
        if resnet_version == "resnet50":
            self.resnet = models.resnet50(pretrained=True)
        elif resnet_version == 'resnet101':
            self.resnet = models.resnet101(pretrained=True)

        # Freeze the parameters of the pre-trained model
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Replace the last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Add new layers for fruit type and color classification
        self.fruit_type_fc = nn.Linear(num_ftrs, num_fruit_types)
        self.color_fc = nn.Linear(num_ftrs, num_colors)

    def forward(self, x):
        features = self.resnet(x)
        fruit_type = self.fruit_type_fc(features)
        color = self.color_fc(features)
        return fruit_type, color