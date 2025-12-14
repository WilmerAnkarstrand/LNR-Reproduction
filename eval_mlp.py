"""
Evaluation Script for MLP models trained on KEEL Datasets
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def parse_keel_dat(filepath):
    """Parse KEEL .dat file format and extract features and labels."""
    features = []
    labels = []
    data_started = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            if line.lower() == '@data':
                data_started = True
                continue
            
            if line.startswith('@'):
                continue
            
            if data_started:
                parts = [p.strip() for p in line.split(',')]
                attribute_count = len(parts) - 1
                try:
                    feature_values = [float(p) for p in parts[:attribute_count]]
                    label = 1 if "positive" in parts[attribute_count].lower() else 0
                    features.append(feature_values)
                    labels.append(label)
                except ValueError:
                    continue

    return np.array(features), np.array(labels)


class MLP(nn.Module):
    """Multi-Layer Perceptron for binary classification."""
    
    def __init__(self, input_dim, hidden_dims=[5, 10, 5]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
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
            
            outputs = model(batch_x).squeeze(-1)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Evaluate MLP model on KEEL dataset')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the saved model (.pth file)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the KEEL dataset (e.g., ecoli1, glass0)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the .dat file (default: Keel_data_sets/{dataset}.dat)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[5, 10, 5],
                        help='Hidden layer dimensions (default: 5 10 5)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation (default: 32)')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Test set proportion (default: 0.3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data path
    data_path = args.data_path if args.data_path else f"Keel_data_sets/{args.dataset}.dat"
    
    # Load and parse data
    print(f"\nLoading data from: {data_path}")
    X, y = parse_keel_dat(data_path)
    print(f"Total samples: {len(X)}")
    
    # Print imbalance info
    n_positive = sum(y)
    n_negative = len(y) - n_positive
    ratio = n_negative / n_positive if n_positive > 0 else float('inf')
    print(f"Positive (minority): {n_positive} ({100*n_positive/len(y):.1f}%)")
    print(f"Negative (majority): {n_negative} ({100*n_negative/len(y):.1f}%)")
    print(f"Imbalance ratio: 1:{ratio:.2f}")
    
    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"\nTest set: {len(X_test)} samples")
    
    # Standardize features (fit on train, transform test)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_test.shape[1]
    model = MLP(input_dim=input_dim, hidden_dims=args.hidden_dims)
    model = model.to(device)
    
    # Load weights
    print(f"\nLoading model from: {args.model}")
    model.load_state_dict(torch.load(args.model, map_location=device))
    
    # Loss function
    num_positives = sum(y)
    num_negatives = len(y) - num_positives
    pos_weight = torch.tensor([num_negatives / num_positives]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Evaluate
    test_loss, test_acc, test_prec, test_rec, test_f1, preds, labels = evaluate(
        model, test_loader, criterion, device
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
    print("="*60)


if __name__ == "__main__":
    main()
