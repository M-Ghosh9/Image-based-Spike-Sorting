#performance.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from model import CustomResNet  # Import the CustomResNet class

def plot_performance_metrics(true_labels, predicted_labels, label_encoder):
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_))

    # F1 Score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"F1 Score: {f1:.2f}")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(true_labels, predicted_labels, pos_label=1)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    # Load model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize LabelEncoder to encode labels
    label_encoder = LabelEncoder()

    # Load preprocessed signals and labels
    signals, labels = torch.load('signals.pt')
    signals = signals.to(device)
    
    # Encode labels
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_labels = torch.tensor(encoded_labels, dtype=torch.long).to(device)

    # Create dataset
    dataset = TensorDataset(signals, encoded_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load the model with the correct number of classes
    num_classes = len(np.unique(encoded_labels.cpu().numpy()))
    model = CustomResNet(num_classes).to(device)

    # Load the state dict from the saved model
    model.load_state_dict(torch.load('spike_classifier.pth', map_location=device))
    model.eval()

    # Predict on the entire dataset
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())

    all_predictions = np.array(all_predictions)

    # Plot performance metrics
    plot_performance_metrics(encoded_labels.cpu().numpy(), all_predictions, label_encoder)

