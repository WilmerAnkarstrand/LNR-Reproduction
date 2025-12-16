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


def make_imbalanced_dataset(X, y, high_classes, medium_classes, low_classes, 
                            medium_target_pct=0.35, low_target_pct=0.07,
                            min_samples_per_class=15, random_seed=42):
    """
    Creates an imbalanced version of the dataset with target percentages.
    
    Args:
        X: Features
        y: Labels
        high_classes: List of class labels for the 'high' frequency group (gets remaining %)
        medium_classes: List of class labels for the 'medium' frequency group
        low_classes: List of class labels for the 'low' frequency group
        medium_target_pct: Target percentage for EACH medium class (0.30-0.40 total for all medium)
        low_target_pct: Target percentage for EACH low class (0.05-0.09 range)
        min_samples_per_class: Minimum samples per class (15 ensures ~10 after 70% train split)
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    
    # First, calculate how many samples we need from each class
    # to achieve target percentages while ensuring minimums
    class_counts = {}
    for label in np.unique(y):
        class_counts[label] = np.sum(y == label)
    
    # Calculate samples to keep for each class type
    samples_to_keep = {}
    
    # For high classes, we'll keep all samples initially
    for label in high_classes:
        samples_to_keep[label] = class_counts[label]
    
    # For medium and low, we need to calculate based on target percentages
    # We'll iterate to find the right balance
    total_high = sum(class_counts[c] for c in high_classes)
    
    # Target: low_target_pct per low class, medium_target_pct per medium class
    # Let's solve: total = high + medium + low
    # medium_pct = medium_samples / total => medium_samples = medium_pct * total
    # We need to find total such that percentages work out
    
    # Using the constraint that high classes keep all samples:
    # total = high + sum(medium) + sum(low)
    # Each medium class should be medium_target_pct of total
    # Each low class should be low_target_pct of total
    
    num_medium = len(medium_classes)
    num_low = len(low_classes)
    
    # high_pct = 1 - (num_medium * medium_target_pct) - (num_low * low_target_pct)
    high_pct = 1.0 - (num_medium * medium_target_pct) - (num_low * low_target_pct)
    
    if high_pct <= 0:
        raise ValueError("Target percentages too high, no room for majority class")
    
    # total = total_high / high_pct
    estimated_total = total_high / high_pct
    
    # Calculate samples for medium and low classes
    for label in medium_classes:
        target_samples = int(estimated_total * medium_target_pct)
        # Ensure we don't exceed available samples and meet minimum
        samples_to_keep[label] = max(min_samples_per_class, 
                                      min(target_samples, class_counts[label]))
    
    for label in low_classes:
        target_samples = int(estimated_total * low_target_pct)
        # Ensure we don't exceed available samples and meet minimum
        samples_to_keep[label] = max(min_samples_per_class, 
                                      min(target_samples, class_counts[label]))
    
    # Handle any classes not specified (treat as high)
    for label in np.unique(y):
        if label not in samples_to_keep:
            samples_to_keep[label] = class_counts[label]
    
    # Now select the samples
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

    # Per-class recall
    per_class_recall = recall_score(all_labels, all_preds, labels=[0,1,2,3,4], average=None, zero_division=0)
    # Head / Medium / Tail
    head_acc = per_class_recall[[0]].mean()           # head class 0
    med_acc  = per_class_recall[[2, 4]].mean()       # medium classes 2 & 4
    tail_acc = per_class_recall[[1, 3]].mean()       # tail classes 1 & 3
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels, head_acc, med_acc, tail_acc




def main(random_seed=42, dataset=None):

        # Create directories if they don't exist
        os.makedirs('keel/mlp', exist_ok=True)
        os.makedirs('keel/mlp_final', exist_ok=True)

        # Configuration
        data_path = f"kaggle_data/{dataset}.csv"
        hidden_dims = [128, 256, 128]  # Increased capacity: 128x256x128
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
        print("Original Dataset Imbalance:")
        print_imbalance_info(y)

        # Make dataset more imbalanced
        # Current: 0:1322, 1:752, 2:521, 3:569, 4:334
        # We keep class 0 as high, 1 and 3 as medium, 2 and 4 as low
        high_classes = [0]
        medium_classes = [2, 4]
        low_classes = [1, 3]
        
        # Target percentages for the imbalanced dataset:
        # - Low classes: ~3.5% each, so ~7% total for both low classes
        # - Medium classes: ~17.5% each (so ~35% total, within 30-40% range)
        # - Majority (high) class: remaining ~58%
        # min_samples_per_class=15 ensures at least 10 samples in training after 70/30 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )
        X_train, y_train = make_imbalanced_dataset(
            X_train, y_train, 
            high_classes=high_classes,
            medium_classes=medium_classes,
            low_classes=low_classes,
            medium_target_pct=0.175,  # Each medium class ~17.5% (total ~35%)
            low_target_pct=0.035,     # Each low class ~3.5% (total ~7%)
            min_samples_per_class=15,  # Ensures ~10+ in training set after split
            random_seed=random_seed
        )
        
        print("New Imbalanced Dataset Information:")
        print_imbalance_info(y_train)
        
        
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
        best_seed = random_seed
        
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _, head_acc, med_acc, tail_acc = evaluate(
                model, test_loader, criterion, device
            )
            
            scheduler.step(val_loss)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch + 1
                best_seed = random_seed
                # Save best model
                torch.save(model.state_dict(), f'keel/mlp/best_mlp_{dataset}_{random_seed}.pth')
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Load best model and final evaluation
        print("\n" + "="*60)
        print(f"Training Complete! Best F1: {best_f1:.4f} at epoch {best_epoch}")
        print("="*60)
        print("best model path:", f'keel/mlp/best_mlp_{dataset}_{best_seed}.pth')
        model.load_state_dict(torch.load(f'keel/mlp/best_mlp_{dataset}_{best_seed}.pth'))
        torch.save(model.state_dict(), f'keel/mlp_final/final_mlp_{dataset}_{best_seed}_{best_f1:.4f}.pth')
        
        test_loss, test_acc, test_prec, test_rec, test_f1, preds, labels, head_acc, med_acc, tail_acc = evaluate(
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

        per_class_prec = precision_score(labels, preds, average=None, zero_division=0)
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    
        return test_acc, test_prec, test_rec, test_f1, per_class_prec, per_class_f1, head_acc, med_acc, tail_acc


if __name__ == "__main__":
    runs = 5
    tot_f1 = 0.0
    f1_scores = []
    tot_prec = 0.0
    prec_scores = []
    
    all_per_class_prec = []
    all_per_class_f1 = []

    overall_acc = []
    all_head_acc = []
    all_med_acc = []
    all_tail_acc = []

    for i in range(runs):
        random_seed = np.random.randint(1, 10000)
        print(f"\n\nRunning experiment with random seed: {random_seed}")
        test_acc, test_prec, test_rec, test_f1, per_class_prec, per_class_f1, test_head_acc, test_med_acc, test_tail_acc = main(random_seed=random_seed, dataset="chronic_disease_dataset")     
        tot_f1 += test_f1
        f1_scores.append(test_f1)
        tot_prec += test_prec
        prec_scores.append(test_prec)
        
        all_per_class_prec.append(per_class_prec)
        all_per_class_f1.append(per_class_f1)

        overall_acc.append(test_acc)
        all_head_acc.append(test_head_acc)
        all_med_acc.append(test_med_acc)
        all_tail_acc.append(test_tail_acc)

    avg_prec = tot_prec / runs
    std_prec = (sum((score - avg_prec) ** 2 for score in prec_scores) / runs) ** 0.5
    print(f"\n\nAverage Precision over {runs} runs: {avg_prec:.4f}")
    print(f"Standard Deviation of Precision: {std_prec:.4f}")
    avg_f1 = tot_f1 / runs
    std_f1 = (sum((score - avg_f1) ** 2 for score in f1_scores) / runs) ** 0.5
    print(f"\n\nAverage F1 over {runs} runs: {avg_f1:.4f}")
    print(f"Standard Deviation of F1: {std_f1:.4f}")

    # Calculate per-class stats
    all_per_class_prec = np.array(all_per_class_prec)
    all_per_class_f1 = np.array(all_per_class_f1)
    
    mean_per_class_prec = np.mean(all_per_class_prec, axis=0)
    std_per_class_prec = np.std(all_per_class_prec, axis=0)
    
    mean_per_class_f1 = np.mean(all_per_class_f1, axis=0)
    std_per_class_f1 = np.std(all_per_class_f1, axis=0)
    
    print("\n" + "="*50)
    print("Per-Class Statistics over {} runs".format(runs))
    print("="*50)
    num_classes = len(mean_per_class_prec)
    for cls in range(num_classes):
        print(f"Class {cls}:")
        print(f"  Precision: {mean_per_class_prec[cls]:.4f} ± {std_per_class_prec[cls]:.4f}")
        print(f"  F1-Score:  {mean_per_class_f1[cls]:.4f} ± {std_per_class_f1[cls]:.4f}")

    mean_overall_acc = np.mean(overall_acc)
    std_overall_acc = np.std(overall_acc)   

    mean_head_acc = np.mean(all_head_acc)
    std_head_acc = np.std(all_head_acc)

    mean_med_acc = np.mean(all_med_acc)
    std_med_acc = np.std(all_med_acc)

    mean_tail_acc = np.mean(all_tail_acc)
    std_tail_acc = np.std(all_tail_acc)
    print("\nOverall Group Accuracies:")
    print(f"  Overall Accuracy:      {mean_overall_acc:.4f} ± {std_overall_acc:.4f}")
    print(f"  Head Class Accuracy:   {mean_head_acc:.4f} ± {std_head_acc:.4f}")
    print(f"  Medium Classes Accuracy: {mean_med_acc:.4f} ± {std_med_acc:.4f}")
    print(f"  Tail Classes Accuracy:   {mean_tail_acc:.4f} ± {std_tail_acc:.4f}")