#performance3544.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, precision_recall_curve, auc, roc_curve, roc_auc_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from modelB3CBAM import EfficientNetCBAMPredictor
from matplotlib.backends.backend_pdf import PdfPages

def plot_detailed_confusion_matrix(true_labels, predicted_labels, label_encoder, class_indices=None, pdf=None):
    cm = confusion_matrix(true_labels, predicted_labels)
    if class_indices is not None:
        cm = cm[class_indices][:, class_indices]
        display_labels = label_encoder.inverse_transform(class_indices)
    else:
        display_labels = label_encoder.classes_
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=display_labels, yticklabels=display_labels)
    plt.title("Detailed Confusion Matrix (Subset of Classes)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if pdf:
        pdf.savefig()
    plt.close()

def plot_class_distribution(labels, label_encoder, pdf=None):
    class_counts = np.bincount(labels)
    sorted_indices = np.argsort(-class_counts)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_indices)), class_counts[sorted_indices], tick_label=label_encoder.inverse_transform(sorted_indices))
    plt.title("Class Frequency Distribution")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    if pdf:
        pdf.savefig()
    plt.close()

def plot_most_confused_classes(true_labels, predicted_labels, label_encoder, top_n=10, pdf=None):
    cm = confusion_matrix(true_labels, predicted_labels)
    np.fill_diagonal(cm, 0)
    flat_cm = cm.flatten()
    sorted_indices = np.argsort(-flat_cm)
    plt.figure(figsize=(10, 6))
    for i in range(top_n):
        idx = sorted_indices[i]
        true_class = idx // len(cm)
        predicted_class = idx % len(cm)
        plt.barh(f'{label_encoder.inverse_transform([true_class])[0]} -> {label_encoder.inverse_transform([predicted_class])[0]}', flat_cm[idx])
    plt.title(f"Top {top_n} Most Confused Classes")
    plt.xlabel("Number of Misclassifications")
    plt.ylabel("True -> Predicted")
    if pdf:
        pdf.savefig()
    plt.close()

def plot_top_k_accuracy(true_labels, probas, k_values=[1, 5, 10], pdf=None):
    accuracies = []
    for k in k_values:
        accuracy = top_k_accuracy_score(true_labels, probas, k=k)
        accuracies.append(accuracy)
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, accuracies, marker='o')
    plt.title("Top-K Accuracy")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.xticks(k_values)
    if pdf:
        pdf.savefig()
    plt.close()

def plot_class_precision_recall(true_labels, probas, label_encoder, class_indices=None, pdf=None):
    plt.figure(figsize=(10, 8))
    for i in class_indices:
        precision, recall, _ = precision_recall_curve(true_labels == i, probas[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{label_encoder.classes_[i]} (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Selected Classes')
    plt.legend(loc="lower left")
    if pdf:
        pdf.savefig()
    plt.close()

def plot_class_roc(true_labels, probas, label_encoder, class_indices=None, pdf=None):
    plt.figure(figsize=(10, 8))
    for i in class_indices:
        fpr, tpr, _ = roc_curve(true_labels == i, probas[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label_encoder.classes_[i]} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Selected Classes')
    plt.legend(loc="lower right")
    if pdf:
        pdf.savefig()
    plt.close()

if __name__ == "__main__":
    # Load model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # Initialize LabelEncoder to encode labels
    label_encoder = LabelEncoder()

    # Load preprocessed signals and labels
    signals, labels = torch.load('signals.pt')
    signals = signals.to(device)
   
    # Encode labels
    print("Encoding labels")
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_labels = torch.tensor(encoded_labels, dtype=torch.long).to(device)

    # Create dataset
    dataset = TensorDataset(signals, encoded_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load the model with the correct number of classes
    num_classes = len(np.unique(encoded_labels.cpu().numpy()))
    model = EfficientNetCBAMPredictor(num_classes).to(device)

    # Load the state dict from the saved model (Ensuring we load the correct model)
    model.load_state_dict(torch.load('checkpointB3CBAM.pth', map_location=device))  # Ensure this matches the save in modelB3CBAM.py
    model.eval()

    # Predict on the entire dataset
    all_predictions = []
    all_probs = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
   
    # Save all plots in a single PDF file
    with PdfPages('performance_report.pdf') as pdf:
        # Plot confusion matrix for all classes
        plot_confusion_matrix(encoded_labels.cpu().numpy(), all_predictions, label_encoder, normalize='true')
       
        # Plot detailed confusion matrix for the top 10 most frequent classes
        most_frequent_classes = np.argsort(-np.bincount(encoded_labels.cpu().numpy()))[:10]
        plot_detailed_confusion_matrix(encoded_labels.cpu().numpy(), all_predictions, label_encoder, class_indices=most_frequent_classes, pdf=pdf)

        # Plot class frequency distribution
        plot_class_distribution(encoded_labels.cpu().numpy(), label_encoder, pdf=pdf)

        # Plot most confused classes
        plot_most_confused_classes(encoded_labels.cpu().numpy(), all_predictions, label_encoder, top_n=10, pdf=pdf)

        # Plot top-k accuracy for k=1, 5, 10
        plot_top_k_accuracy(encoded_labels.cpu().numpy(), all_probs, k_values=[1, 5, 10], pdf=pdf)

        # Plot precision-recall curve for the top 5 most frequent classes
        plot_class_precision_recall(encoded_labels.cpu().numpy(), all_probs, label_encoder, class_indices=most_frequent_classes[:5], pdf=pdf)

        # Plot ROC curve for the top 5 most frequent classes
        plot_class_roc(encoded_labels.cpu().numpy(), all_probs, label_encoder, class_indices=most_frequent_classes[:5], pdf=pdf)

    # Additional Metrics
    print("Classification Report:")
    print(classification_report(encoded_labels.cpu().numpy(), all_predictions, target_names=label_encoder.classes_))

    f1 = f1_score(encoded_labels.cpu().numpy(), all_predictions, average='weighted')
    print(f"F1 Score: {f1:.2f}")

    top_5_accuracy = top_k_accuracy_score(encoded_labels.cpu().numpy(), all_probs, k=5)
    print(f"Top-5 Accuracy: {top_5_accuracy:.2f}")
