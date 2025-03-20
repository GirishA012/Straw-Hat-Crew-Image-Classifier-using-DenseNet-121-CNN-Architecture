import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms, datasets
from torchvision.models import DenseNet121_Weights
import os

def train_model():
    # ✅ Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ✅ Define data path
    data_dir = "augmented_data"

    # ✅ Define transformations for training and testing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    # ✅ Load dataset
    print("Loading dataset...")
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes
    print(f"Dataset Loaded! Found {len(dataset)} images across {len(class_names)} classes.")

    # ✅ Split dataset: 80% train, 20% test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # ✅ Use DataLoader for batch processing
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ✅ Load pre-trained DenseNet-121 model
    print("Loading DenseNet-121 model...")
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(class_names))  # Modify classifier for our dataset

    # ✅ Move model to GPU if available
    model = model.to(device)

    # ✅ Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ Training loop with batch processing
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch+1}/{num_epochs} training...")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # ✅ Testing phase (Using 20% test dataset)
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}% - "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # ✅ Save the trained model
    os.makedirs("models", exist_ok=True)
    model_path = "models/densenet121.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path} ✅")

# ✅ Windows fix: Use "if __name__ == '__main__'":
if __name__ == "__main__":
    train_model()
