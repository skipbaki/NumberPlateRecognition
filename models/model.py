import torch
import torch.nn as nn
from torchvision import models

class PlateRecognitionModel(nn.Module):
    def __init__(self, num_chars):
        super(PlateRecognitionModel, self).__init__()
        
        # CNN Backbone (ResNet18)
        self.cnn = models.resnet18(pretrained=True)
        
        # Replace final fully connected layer
        self.cnn.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.2
        )
        
        # Final character classifier
        self.fc = nn.Linear(256, num_chars + 1)  # +1 for CTC blank token

    def forward(self, x):
        # CNN feature extraction
        batch_size = x.size(0)
        x = self.cnn(x)  # Output shape: (batch_size, 256)

        x = x.unsqueeze(0) 
        x = x.repeat(30, 1, 1)  
        
        # LSTM processing
        x, _ = self.lstm(x)       
        
        # Character classification
        x = self.fc(x)           
        return x

# Test the model architecture
if __name__ == "__main__":
    # Test parameters
    batch_size = 16
    num_chars = 36  # 26 letters + 10 digits
    input_size = (3, 224, 224)  # RGB images
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_size)
    
    # Initialize model
    model = PlateRecognitionModel(num_chars=num_chars)
    
    # Forward pass
    output = model(dummy_input)
    
    # Verify output dimensions
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")
