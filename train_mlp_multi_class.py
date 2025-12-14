"""
MLP Training Script for KEEL Ecoli1 Dataset
Multi-class classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import sys


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

def print_imbalance_info(y):
    """Print imbalance information for the dataset."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    print("\n" + "="*50)
    print("Dataset Imbalance Information")
    print("="*50)
    print(f"Total samples:    {total}")
    for label, count in zip(unique, counts):
        print(f"Class {int(label)}: {count} ({100*count/total:.1f}%)")
    print("="*50 + "\n")
    
    return counts


def make_imbalanced_dataset(X, y, high_classes, medium_classes, low_classes, medium_ratio, low_ratio, random_seed=42):
    """
    Creates an imbalanced version of the dataset.
    
    Args:
        X: Features
        y: Labels
        high_classes: List of class labels for the 'high' frequency group (kept at 100%)
        medium_classes: List of class labels for the 'medium' frequency group
        low_classes: List of class labels for the 'low' frequency group
        medium_ratio: Ratio of samples to keep for medium classes (0 < ratio <= 1)
        low_ratio: Ratio of samples to keep for low classes (0 < ratio <= 1)
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    indices_to_keep = []
    
    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        n_samples = len(label_indices)
        
        if label in high_classes:
            indices_to_keep.extend(label_indices)
        elif label in medium_classes:
            n_keep = max(1, int(n_samples * medium_ratio))
            selected = np.random.choice(label_indices, n_keep, replace=False)
            indices_to_keep.extend(selected)
        elif label in low_classes:
            n_keep = max(1, int(n_samples * low_ratio))
            selected = np.random.choice(label_indices, n_keep, replace=False)
            indices_to_keep.extend(selected)
        else:
            # If class not specified, keep it (treat as high)
            indices_to_keep.extend(label_indices)
            
    indices_to_keep = np.array(indices_to_keep)
    np.random.shuffle(indices_to_keep)
    
    return X[indices_to_keep], y[indices_to_keep]


class MLP(nn.Module):
    """Multi-Layer Perceptron for binary classification."""
    
    def __init__(self, input_dim, hidden_dims=[5, 10, 5], num_classes=5):
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
        #layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            #print so i can debug the batch shapes

            
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


def main(random_seed=42, dataset=None):


        # Configuration
        data_path = f"kaggle_data/{dataset}.csv"
        hidden_dims = [5, 10, 5]  # Hidden layer shape 5x10x5
        learning_rate = 0.001  # Adam learning rate
        batch_size = 32
        num_epochs = 200  # 2000 for KEEL data, 800 for synthetic data
        test_size = 0.3  # 30% test size
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load and parse data
        print(f"\nLoading data from: {data_path}")
        X, y = parse_kaggle_dat(data_path)
        print(f"First 5 samples of X:\n{X[0:5]}")
        print(f"First 5 labels of y:\n{y[0:5]}")
        # Print imbalance information
        print("Original Dataset Imbalance:")
        print_imbalance_info(y)

        # Make dataset more imbalanced
        # Current: 0:1322, 1:752, 2:521, 3:569, 4:334
        # We keep class 0 as high, 1 and 3 as medium, 2 and 4 as low
        high_classes = [0]
        medium_classes = [1, 3]
        low_classes = [2, 4]
        
        # medium_ratio=0.25 means we keep 25% of the original samples for medium classes
        # low_ratio=0.02 means we keep 2% of the original samples for low classes
        X, y = make_imbalanced_dataset(
            X, y, 
            high_classes=high_classes,
            medium_classes=medium_classes,
            low_classes=low_classes,
            medium_ratio=0.25,
            low_ratio=0.02,
            random_seed=random_seed
        )
        
        print("New Imbalanced Dataset Information:")
        print_imbalance_info(y)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_dim = X_train.shape[1]
        model = MLP(input_dim=input_dim, hidden_dims=hidden_dims)
        model = model.to(device)
        print(f"\nModel architecture:")
        print(model)
        
        # Calculate class weights to handle imbalance
        # Cast to int because labels are loaded as floats
        y_train_int = y_train.astype(int)
        class_counts = np.bincount(y_train_int)
        # Standard formula for balanced weights: n_samples / (n_classes * n_samples_j)
        num_classes_present = len(np.unique(y_train_int))
        total_samples = len(y_train_int)
        class_weights = total_samples / (num_classes_present * np.maximum(class_counts, 1))
        
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        print(f"\nClass Weights: {class_weights}")

        # Loss function
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        
        # Training loop
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        best_f1 = 0.0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
                model, test_loader, criterion, device
            )
            
            scheduler.step(val_loss)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch + 1
                # Save best model
                torch.save(model.state_dict(), f'keel/mlp/best_mlp_{dataset}.pth')
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Load best model and final evaluation
        print("\n" + "="*60)
        print(f"Training Complete! Best F1: {best_f1:.4f} at epoch {best_epoch}")
        print("="*60)
        
        model.load_state_dict(torch.load(f'keel/mlp/best_mlp_{dataset}.pth'))
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
    
        return test_acc, test_prec, test_rec, test_f1


if __name__ == "__main__":
    runs = 1
    tot_f1 = 0.0
    f1_scores = []
    for i in range(runs):
        random_seed = np.random.randint(1, 10000)
        print(f"\n\nRunning experiment with random seed: {random_seed}")
        test_acc, test_prec, test_rec, test_f1 = main(random_seed=random_seed, dataset="chronic_disease_dataset")     
        tot_f1 += test_f1
        f1_scores.append(test_f1)
    avg_f1 = tot_f1 / runs
    std_f1 = (sum((score - avg_f1) ** 2 for score in f1_scores) / runs) ** 0.5
    print(f"\n\nAverage F1 over 10 runs: {avg_f1:.4f}")
    print(f"Standard Deviation of F1: {std_f1:.4f}")