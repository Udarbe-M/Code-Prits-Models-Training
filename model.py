import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extraction using parallel convolutional branches
    """
    def __init__(self, in_channels):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # Branch 1: Small receptive field (3x3)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: Medium receptive field (5x5)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: Large receptive field (7x7)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 conv to reduce dimensions after concatenation
        self.fusion = nn.Sequential(
            nn.Conv2d(192, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # Concatenate multi-scale features
        concat = torch.cat([b1, b2, b3], dim=1)
        
        # Fuse features
        out = self.fusion(concat)
        return out


class AttentionModule(nn.Module):
    """
    Spatial attention mechanism to focus on important regions
    """
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Generate attention map
        att = self.conv1(x)
        att = F.relu(att)
        att = self.conv2(att)
        att = self.sigmoid(att)
        
        # Apply attention
        out = x * att
        return out


class CNNEncoder(nn.Module):
    """
    CNN backbone for feature extraction with multi-scale features and attention
    """
    def __init__(self, img_height, num_channels=1):
        super(CNNEncoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Multi-scale block 1 + Attention
        self.multiscale1 = MultiScaleFeatureExtractor(64)
        self.attention1 = AttentionModule(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale block 2 + Attention
        self.multiscale2 = MultiScaleFeatureExtractor(128)
        self.attention2 = AttentionModule(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale block 3 + Attention
        self.multiscale3 = MultiScaleFeatureExtractor(256)
        self.attention3 = AttentionModule(256)
        
        # Conv block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Calculate final height after pooling operations
        self.final_height = img_height // 8  # 3 pooling layers with stride 2
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        
        # Multi-scale + Attention block 1
        x = self.multiscale1(x)
        x = self.attention1(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        
        # Multi-scale + Attention block 2
        x = self.multiscale2(x)
        x = self.attention2(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        
        # Multi-scale + Attention block 3
        x = self.multiscale3(x)
        x = self.attention3(x)
        
        # Final conv
        x = self.conv4(x)
        
        return x


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM layer
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.linear(recurrent)
        return output


class AttentionGuidedMSFCRNN(nn.Module):
    """
    Complete Attention-Guided Multi-Scale Feature Convolutional Recurrent Neural Network
    for handwritten text recognition
    """
    def __init__(self, img_height, img_channels, num_classes, hidden_size=256):
        super(AttentionGuidedMSFCRNN, self).__init__()
        
        self.img_height = img_height
        self.num_classes = num_classes
        
        # CNN Encoder with multi-scale features and attention
        self.cnn = CNNEncoder(img_height, img_channels)
        
        # Calculate RNN input size based on CNN output
        # After CNN, we'll have: batch x 512 x (height/8) x width
        self.rnn_input_size = 512 * self.cnn.final_height
        
        # Recurrent layers
        self.rnn1 = BidirectionalLSTM(self.rnn_input_size, hidden_size, hidden_size)
        self.rnn2 = BidirectionalLSTM(hidden_size, hidden_size, num_classes)
    
    def forward(self, x):
        # CNN feature extraction
        # Input: batch x channels x height x width
        conv_features = self.cnn(x)
        
        # Reshape for RNN
        # conv_features: batch x 512 x reduced_height x width
        batch, channels, height, width = conv_features.size()
        
        # Permute to: batch x width x (channels * height)
        conv_features = conv_features.permute(0, 3, 1, 2)
        conv_features = conv_features.contiguous().view(batch, width, -1)
        
        # RNN sequence modeling
        rnn_out = self.rnn1(conv_features)
        output = self.rnn2(rnn_out)
        
        # Output: batch x width x num_classes
        return output


def create_model(img_height=32, img_channels=1, num_classes=80, hidden_size=256):
    """
    Factory function to create the model
    
    Args:
        img_height: Height of input images
        img_channels: Number of channels (1 for grayscale, 3 for RGB)
        num_classes: Number of character classes (including blank for CTC)
        hidden_size: Hidden size for LSTM layers
    
    Returns:
        model: AttentionGuidedMSFCRNN model
    """
    model = AttentionGuidedMSFCRNN(
        img_height=img_height,
        img_channels=img_channels,
        num_classes=num_classes,
        hidden_size=hidden_size
    )
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(img_height=32, img_channels=1, num_classes=80)
    model = model.to(device)
    
    # Test with random input
    batch_size = 4
    img_channels = 1
    img_height = 32
    img_width = 128
    
    dummy_input = torch.randn(batch_size, img_channels, img_height, img_width).to(device)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch={batch_size}, width={img_width}, num_classes=80)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")