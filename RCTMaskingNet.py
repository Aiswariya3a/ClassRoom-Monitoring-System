import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

# --- Channel Cross Transformer Masking (CCT) Block ---
class CCTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=16):
        super(CCTBlock, self).__init__()
        self.patch_size = patch_size
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x):
        # Input shape: [Batch, Channels, Height, Width]
        B, C, H, W = x.shape
        patch_size = self.patch_size
        
        # Convert input to patch sequences
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.reshape(B, C, -1, patch_size * patch_size)  # [B, C, Patches, Patch_Size^2]
        
        # Compute Query, Key, Value
        Q = self.q_proj(patches)
        K = self.k_proj(patches)
        V = self.v_proj(patches)
        
        # Cross-Attention Mechanism
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(C, dtype=torch.float32))
        attention_weights = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_weights, V)
        
        # MLP and residual connection
        attention_output = self.mlp(attention_output) + patches
        return attention_output.reshape(B, C, H, W)  # Reshape to original dimensions

# --- Channel Cross-Attention (CCA) Block ---
class CCABlock(nn.Module):
    def __init__(self, in_channels):
        super(CCABlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Global Average Pooling
        channel_weights = self.global_avg_pool(x)
        channel_weights = self.channel_attention(channel_weights)
        return x * channel_weights

# --- RCTMaskingNet ---
class RCTMaskingNet(nn.Module):
    def __init__(self, num_classes=5):
        super(RCTMaskingNet, self).__init__()
        # Load pre-trained ResNet34
        backbone = resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            *list(backbone.layer1),
        )
        self.residual_layers = nn.ModuleList([
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        ])
        
        # Add CCT and CCA Blocks
        self.cct_blocks = nn.ModuleList([
            CCTBlock(64, 128),  # Adjust input/output channels as per layer
            CCTBlock(128, 256),
            CCTBlock(256, 512)
        ])
        self.cca_blocks = nn.ModuleList([
            CCABlock(64),
            CCABlock(128),
            CCABlock(256),
            CCABlock(512)
        ])
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Initial feature extraction
        x = self.feature_extractor(x)  # Output: [B, 64, H/4, W/4]
        
        # Process through Residual + CCT + CCA blocks
        for i in range(len(self.residual_layers)):
            x = self.residual_layers[i](x)
            x = self.cct_blocks[i](x)
            x = self.cca_blocks[i](x)
        
        # Final attention and classification
        x = self.classifier(x)
        return x

# --- Model Initialization and Summary ---
if __name__ == "__main__":
    model = RCTMaskingNet(num_classes=5)
    print(model)
    
    # dummy_input = torch.randn(1, 3, 224, 224)  # Batch size: 1, Image size: 224x224
    # output = model(dummy_input)
    # print(f"Output shape: {output.shape}")  # Expected: [1, num_classes]
