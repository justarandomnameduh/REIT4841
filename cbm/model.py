import torch
import torch.nn as nn
import torchvision.models as models

class ConceptBottleneckModel(nn.Module):
    """
    Concept Bottleneck Model (CBM) using a ResNet backbone.\
    
    Args:
        num_classes (int): The number of target classes for final prediction
        num_concepts (int): The number of concept outputs in the bottleneck layer
        pretrained (bool): Whether to use ImageNet pre-trained weights
    """

    def __init__(self, num_classes, num_concepts, pretrained=True):
        super(ConceptBottleneckModel, self).__init__()
        self.num_concepts = num_concepts

        # Load ResNet50 backbone
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2

            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)

        num_ftrs = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        self.concept_bottleneck = nn.Linear(num_ftrs, num_concepts)

        self.final_classifier = nn.Linear(num_concepts, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            tuple: Tuple containing:
                - concept_outputs (torch.Tensor): Output from the concept bottleneck layer
                - final_outputs (torch.Tensor): Final class predictions
        """
        features = self.backbone(x)
        concept_logits = self.concept_bottleneck(features)
        final_logits = self.final_classifier(concept_logits)
        return final_logits, concept_logits