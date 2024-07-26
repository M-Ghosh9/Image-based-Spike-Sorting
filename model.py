#model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt

class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def compute_class_weights(labels):
    class_sample_count = np.unique(labels, return_counts=True)[1]
    class_weights = 1. / class_sample_count
    return class_weights

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return loss, accuracy

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    print(f"Checkpoint saved as '{filename}'")

def load_checkpoint(filename='checkpoint.pth', model=None, optimizer=None):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        if model is not None:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print(f"Error loading model state_dict: {e}")
                # Handle size mismatch by adjusting the model's fc layer
                num_classes = model.resnet.fc.out_features
                model.resnet.fc = nn.Linear(model.resnet.fc.in_features, num_classes)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_accuracy = checkpoint['best_val_accuracy']
        return epoch, best_val_accuracy
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, 0

def predict(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_probabilities)

def plot_class_distribution(labels, filename='class_distribution.png', title='Class Distribution'):
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(12, 6))
    plt.bar(unique_labels, counts, align='center')
    plt.xlabel('Class Labels')
    plt.ylabel('Counts')
    plt.title(title)
    plt.savefig(filename)  # Save plot as an image file
    plt.close()  # Close the plot to free memory

if __name__ == "__main__":
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load preprocessed images and labels
        print("Loading images and labels...")
        images, image_labels = torch.load('images.pt')
        print(f"Images shape: {images.shape}")

        # Ensure image_labels is in a usable format
        if isinstance(image_labels, list):
            print("Converting image_labels from list to numpy array...")
            image_labels = np.array(image_labels)
        elif isinstance(image_labels, torch.Tensor):
            image_labels = image_labels.numpy()
        elif not isinstance(image_labels, np.ndarray):
            raise TypeError(f"Unexpected type for image_labels: {type(image_labels)}")

        print(f"Labels shape: {image_labels.shape}")

        # Encode labels
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(image_labels)
        encoded_labels = torch.tensor(encoded_labels, dtype=torch.long).to(device)
        print(f"Number of unique classes: {len(np.unique(encoded_labels.cpu().numpy()))}")

        # Plot class distribution before handling imbalance
        print("Plotting class distribution before handling imbalance...")
        plot_class_distribution(image_labels, filename='class_distribution_before.png', title='Class Distribution Before Handling Imbalance')

        # Compute class weights
        print("Computing class weights...")
        class_weights = compute_class_weights(encoded_labels.cpu().numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        # Preprocessing transformation including resize to 224x224
        print("Creating preprocessing transformation...")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Create dataset
        print("Creating dataset...")
        dataset = TensorDataset(images, encoded_labels)

        # Move encoded_labels to CPU before converting to NumPy array
        encoded_labels_cpu = encoded_labels.cpu().numpy()

        # Get the number of unique classes
        num_classes = len(np.unique(encoded_labels_cpu))
        print(f"Number of classes for model: {num_classes}")

        # Split dataset into training and validation sets (80-20)
        print("Splitting dataset into training and validation sets...")
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print(f"Training set size: {train_size}, Validation set size: {val_size}")

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Initialize model, criterion, and optimizer
        print("Initializing model, criterion, and optimizer...")
        model = CustomResNet(num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Load checkpoint if available
        print("Loading checkpoint...")
        start_epoch, best_val_accuracy = load_checkpoint('checkpoint.pth', model, optimizer)
        
        # Training loop
        num_epochs = 50
        print("Starting training loop...")
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / len(train_dataloader)
            train_accuracy = 100. * correct / total

            val_loss, val_accuracy = evaluate_model(model, val_dataloader, criterion, device)

            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            # Save checkpoint if validation accuracy improves
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_accuracy': best_val_accuracy
                })

        # Final save
        print("Saving final model...")
        torch.save(model.state_dict(), 'spike_classifier.pth')
        print("Final model saved as 'spike_classifier.pth'")

        # Predict and convert images to spikes
        print("Loading final model for predictions...")
        model.load_state_dict(torch.load('spike_classifier.pth'))
        model.to(device)
        
        # Create DataLoader for predictions (same images used for training here)
        test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Predict on the entire dataset
        print("Predicting on the dataset...")
        predictions, probabilities = predict(model, test_dataloader, device)

        # Convert predictions to class labels
        predicted_labels = label_encoder.inverse_transform(predictions)
        
        # Save predictions to file
        print("Saving predictions and probabilities...")
        np.save('predicted_labels.npy', predicted_labels)
        np.save('probabilities.npy', probabilities)
        
        print("Predictions and probabilities saved.")
        
        # Plot class distribution after handling imbalance (if needed)
        print("Plotting class distribution after handling imbalance...")
        plot_class_distribution(encoded_labels.cpu().numpy(), filename='class_distribution_after.png', title='Class Distribution After Handling Imbalance')

    except Exception as e:
        print(f"An error occurred: {e}")


