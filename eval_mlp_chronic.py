"""
Evaluation Script for Trained MLP Model on Chronic Disease Dataset
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import sys
import argparse

def parse_kaggle_dat(filepath):
    """Parse kaggle .csv file and return features and labels.
       Healthy, diabetes, cardiovascular disorder, cancer, and multi-condition cases.
       As 0,1,2,3,4 labels."""
    features = []
    labels = []
    data_started = False

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip header row
            if line_num == 0:
                data_started = True
                continue

            # Parse data lines
            if data_started:
                parts = [p.strip() for p in line.split(',')]
                attribute_count = len(parts) - 1  # Last part is the label
                try:
                    feature_values = [float(p) for p in parts[:attribute_count]]
                    label = float(parts[attribute_count])
                    features.append(feature_values)
                    labels.append(label)
                except ValueError:
                    continue

    return np.array(features), np.array(labels)

def print_imbalance_info(y, name="Dataset"):
    """Print imbalance information for the dataset."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    print("\n" + "="*50)
    print(f"{name} Imbalance Information")
    print("="*50)
    print(f"Total samples:    {total}")
    for label, count in zip(unique, counts):
        print(f"Class {int(label)}: {count} ({100*count/total:.1f}%)")
    print("="*50 + "\n")
    
    return counts

def make_imbalanced_dataset(X, y, high_classes, medium_classes, low_classes, 
                            medium_target_pct=0.35, low_target_pct=0.07,
                            min_samples_per_class=15, random_seed=42):
    """
    Creates an imbalanced version of the dataset with target percentages.
    """
    np.random.seed(random_seed)
    
    class_counts = {}
    for label in np.unique(y):
        class_counts[label] = np.sum(y == label)
    
    samples_to_keep = {}
    
    for label in high_classes:
        samples_to_keep[label] = class_counts[label]
    
    total_high = sum(class_counts[c] for c in high_classes)
    
    num_medium = len(medium_classes)
    num_low = len(low_classes)
    
    high_pct = 1.0 - (num_medium * medium_target_pct) - (num_low * low_target_pct)
    
    if high_pct <= 0:
        raise ValueError("Target percentages too high, no room for majority class")
    
    estimated_total = total_high / high_pct
    
    for label in medium_classes:
        target_samples = int(estimated_total * medium_target_pct)
        samples_to_keep[label] = max(min_samples_per_class, 
                                      min(target_samples, class_counts[label]))
    
    for label in low_classes:
        target_samples = int(estimated_total * low_target_pct)
        samples_to_keep[label] = max(min_samples_per_class, 
                                      min(target_samples, class_counts[label]))
    
    for label in np.unique(y):
        if label not in samples_to_keep:
            samples_to_keep[label] = class_counts[label]
    
    indices_to_keep = []
    
    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        n_keep = samples_to_keep[label]
        
        if n_keep >= len(label_indices):
            indices_to_keep.extend(label_indices)
        else:
            selected = np.random.choice(label_indices, n_keep, replace=False)
            indices_to_keep.extend(selected)
            
    indices_to_keep = np.array(indices_to_keep)
    np.random.shuffle(indices_to_keep)
    
    return X[indices_to_keep], y[indices_to_keep]

class MLP(nn.Module):
    """Multi-Layer Perceptron for binary classification."""
    
    def __init__(self, input_dim, hidden_dims=[128, 256, 128], num_classes=5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # Output layer for multi-class classification
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description='Evaluate MLP on Chronic Disease Dataset')
    parser.add_argument('--model_path', type=str, default=r'keel\mlp_final\final_mlp_chronic_disease_dataset_7110_0.2555.pth', help='Path to the trained model')
    parser.add_argument('--seed', type=int, default=7110, help='Random seed')
    parser.add_argument('--dataset', type=str, default='chronic_disease_dataset', help='Dataset name')
    args = parser.parse_args()

    # Parameters matching the trained model
    model_path = args.model_path
    dataset = args.dataset
    random_seed = args.seed
    
    print(f"Evaluating model: {model_path}")
    print(f"Using random seed: {random_seed}")
    
    # Configuration (must match training)
    data_path = f"kaggle_data/{dataset}.csv"
    hidden_dims = [128, 256, 128]
    batch_size = 32
    test_size = 0.3
    
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and parse data
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from: {data_path}")
    X, y = parse_kaggle_dat(data_path)
    
    # Reproduce the exact data split and processing
    high_classes = [0]
    medium_classes = [2, 4]
    low_classes = [1, 3]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    # Apply imbalance to training set (needed to reproduce the scaler fit)
    X_train, y_train = make_imbalanced_dataset(
        X_train, y_train, 
        high_classes=high_classes,
        medium_classes=medium_classes,
        low_classes=low_classes,
        medium_target_pct=0.175,
        low_target_pct=0.035,
        min_samples_per_class=15,
        random_seed=random_seed
    )
    
    print_imbalance_info(y_test, "Test Set")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # Fit on train
    X_test = scaler.transform(X_test)       # Transform test
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims)
    model = model.to(device)
    
    # Load trained weights
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    print("Loading model weights...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Loss function (needed for evaluate function signature, though we care mostly about metrics)
    # We need class weights to match the criterion used in evaluate, although for metrics it doesn't matter much
    # except for the loss value returned.
    y_train_int = y_train.astype(int)
    class_counts = np.bincount(y_train_int)
    num_classes_present = len(np.unique(y_train_int))
    total_samples = len(y_train_int)
    class_weights = total_samples / (num_classes_present * np.maximum(class_counts, 1))
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Evaluate
    print("\nEvaluating...")
    test_loss, test_acc, test_prec, test_rec, test_f1, preds, labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print("\nFinal Test Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(labels, preds, zero_division=0))

if __name__ == "__main__":
    main()
