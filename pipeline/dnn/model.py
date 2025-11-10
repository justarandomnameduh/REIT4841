import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_V2_S_Weights


class EfficientNetV2(nn.Module):
    """
    EfficientNetV2 model for both classification and embedding extraction.
    Uses ImageNet pre-trained weights by default for better performance.
    """
    
    def __init__(self, num_outputs, dropout=0.2, pretrained=True):
        """
        Args:
            num_outputs: Number of output classes or concepts
            dropout: Dropout rate for the classifier
            pretrained: If True, use ImageNet pre-trained weights (default: True)
        """
        super(EfficientNetV2, self).__init__()
        
        if pretrained:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_v2_s(weights=weights)
        else:
            self.backbone = models.efficientnet_v2_s(weights=None)
        
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(num_features, num_outputs)
        )
        
        self.num_features = num_features
        self.num_outputs = num_outputs
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_embedding(self, x):
        """
        Extract feature embeddings before the final classification layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature embeddings
        """
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return features
    
    def load_pretrained_backbone(self, pretrained_path, num_outputs_new):
        """
        Load pre-trained weights and replace the final classification layer.
        
        Args:
            pretrained_path: Path to pre-trained model weights
            num_outputs_new: Number of outputs for the new task
        """
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        state_dict = {}
        for k, v in checkpoint.items():
            if 'classifier.1' not in k:
                state_dict[k] = v
        
        self.backbone.load_state_dict(state_dict, strict=False)
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.num_features, num_outputs_new)
        )
        self.num_outputs = num_outputs_new
