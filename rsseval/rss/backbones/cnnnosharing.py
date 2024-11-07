import torch
import torch.nn as nn
import torch.nn.functional as F

class IndividualMNISTCNN(nn.Module):
    def __init__(self, outclasses):
        super(IndividualMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, outclasses)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNNoSharing(nn.Module):
    def __init__(self, num_images, nout, output1, output2):
        super(CNNNoSharing, self).__init__()
        self.cnn_modules = nn.ModuleList([IndividualMNISTCNN(nout) for _ in range(num_images)])
        self.classifier = nn.Sequential(
            # nn.Linear(self.n_facts * self.n_images, self.n_facts * self.n_images), nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        outputs = [cnn(xi) for cnn, xi in zip(self.cnn_modules, x)]
        stacked_output = torch.cat(outputs, dim=1)  # Concatenate the outputs along the feature dimension
        return self.classifier(stacked_output)
    
class CBMNoSharing(nn.Module):
    def __init__(self, num_images, nout):
        super(CBMNoSharing, self).__init__()
        self.cnn_modules = nn.ModuleList([IndividualMNISTCNN(nout) for _ in range(num_images)])

    def forward(self, x):
        outputs = [cnn(xi) for cnn, xi in zip(self.cnn_modules, x)]
        stacked_output = torch.stack(outputs, dim=1)  # Concatenate the outputs along the feature dimension
        
        return stacked_output

class MNISTLCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MNISTLCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # Output: [16, 28, 112]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # Output: [32, 14, 56]
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # Output: [64, 7, 28]
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 28, 128)  # Flattened output from conv layers
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)  # Final output layer
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):

        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))  # [16, 28, 112] -> [16, 14, 56]
        x = self.pool(F.relu(self.conv2(x)))  # [32, 14, 56] -> [32, 7, 28]
        x = F.relu(self.conv3(x))             # [64, 7, 28]
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64 * 7 * 28)
        
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Final output layer
        x = self.fc3(x)
        
        return self.softmax(x)

class MNMNISTCNN(nn.Module):
    def __init__(self, num_classes=2, num_images=4):
        super(MNMNISTCNN, self).__init__()
        
        # Define the number of images concatenated
        self.num_images = num_images
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # Output: [16, 28, 28*num_images]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # Output: [32, 14, 14*num_images]
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # Output: [64, 7, 7*num_images]
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7 * num_images, 128)  # Flattened output from conv layers
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)  # Final output layer
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))  # [16, 28, 28*num_images] -> [16, 14, 14*num_images]
        x = self.pool(F.relu(self.conv2(x)))  # [32, 14, 14*num_images] -> [32, 7, 7*num_images]
        x = F.relu(self.conv3(x))             # [64, 7, 7*num_images]
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64 * 7 * 7 * self.num_images)
        
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Final output layer
        x = self.fc3(x)
        
        return self.sigmoid(x)

if __name__ == "__main__":
    # Assuming you have 3 images to process, each of size [1, 28, 28]
    num_images = 3
    n_out = 10
    out1 = 5
    out2 = 2
    model = MNISTLCNN()
    
    # Create dummy data for 3 images (batch size 1, each image of size 1x28x28)
    images = [torch.randn(1, 1, 28, 112) for _ in range(num_images)]
    
    # Forward pass
    logits = model(torch.randn(1, 1, 28, 112))
    print(logits.shape)  # Expected output shape: [1, 2]