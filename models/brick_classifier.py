'''
Different classifiers are defined to classify detected bricks into different classes. 
Note: Output classes identify different types of bricks or different attributes based on which the bricks are being sorted.
'''

import torch
import torch.nn as nn
import torchvision.models as models

class SimpleClassifier(nn.Module):
    """
    A simple fully connected neural network for classification.
    The architecture can be modified according to the data size a problem complexity.

    Parameters:
    num_classes (int): Number of output classes for the classifier. 

    """
    def __init__(self, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(3 * 64 * 64, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, num_classes)  # Second fully connected layer

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, 3, 64, 64).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = torch.relu(self.fc1(x))  # Apply ReLU activation after first layer
        x = self.fc2(x)  # Apply second layer
        return x

class ResNetClassifier(nn.Module):
    """
    A ResNet-based classifier with a modified final layer for custom number of classes.

    Parameters:
    num_classes (int): Number of output classes for the classifier.
    """
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)  # Load pretrained ResNet50 model
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Modify final layer for classification

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.model(x)
    

class AttributeClassifier(nn.Module):
    """
    A ResNet-based classifier with additional attributes.

    Parameters:
    num_classes (int): Number of output classes for the classifier.
    attr_dim (int): Dimension of the additional attributes.
    """
    def __init__(self, num_classes, attr_dim):
        super(AttributeClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # Load pretrained ResNet50 model
        self.resnet.fc = nn.Identity()  # Replace the final layer with identity to get features
        self.fc = nn.Linear(self.resnet.fc.in_features + attr_dim, num_classes)  # New final layer including attributes

    def forward(self, x, attrs):
        """
        Forward pass through the network.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).
        attrs (torch.Tensor): Additional attributes tensor of shape (batch_size, attr_dim).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        features = self.resnet(x)  # Get features from ResNet
        combined = torch.cat((features, attrs), dim=1)  # Concatenate features with attributes
        out = self.fc(combined)  # Pass through final classification layer
        return out


# Example usage
if __name__ == "__main__":
    simple_model = SimpleClassifier(num_classes=10)
    resnet_model = ResNetClassifier(num_classes=10)
    attr_resnet_model = AttributeClassifier(num_classes=10, attr_dim=2)
    print(simple_model)
    print(resnet_model)
    print(attr_resnet_model)
