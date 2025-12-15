import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import copy

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

def lnr(model, x_train, y_train, device, t_flip):
    """
    Train ONLY the last layer using LNR logic.
    Assumes model is already pre-trained and hidden layers are frozen.
    """
    model.eval()

    # Convert numpy arrays to tensors efficiently
    if not isinstance(x_train, torch.Tensor):
        x_train = torch.tensor(x_train, device=device, dtype=torch.float32)
    else:
        x_train = x_train.to(device)
    
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, device=device, dtype=torch.long)
    else:
        y_train = y_train.to(device)

    with torch.no_grad():
        logits = model(x_train)
        Q = nn.Softmax(dim=1)(logits)
        mean = Q.mean(dim=0)
        std = Q.std(dim=0)

    # Count samples per class
    num_classes = int(y_train.max()) + 1
    class_count = torch.tensor([(y_train == i).sum().item() for i in range(num_classes)], device=device, dtype=torch.float32)

    # θC ← 1 − minMax(NC) where NC counts the samples of each class
    theta_c = 1 - (class_count - class_count.min()) / (class_count.max() - class_count.min())

    # Z_score_x
    Z_score_x = (Q - mean) / (std + 1e-8)
    
    # Fix broadcasting: [1, C] - [B, 1] -> [B, C]
    flip_strength = torch.clamp(theta_c.unsqueeze(0) - theta_c[y_train].unsqueeze(1), min=0)

    #Class wise flip strength: θc=y
    class_fliprate = torch.clamp(torch.tanh(Z_score_x - t_flip), min=0) * flip_strength
    bernoulli_samples = torch.bernoulli(class_fliprate)

    # Create a copy for relabeling
    y_train_flipped = y_train.clone()
    total_flips = 0

    # For each sample, check if any class got a 1 (flip should happen)
    for i in range(len(y_train)):
        if bernoulli_samples[i].sum() > 0:  # U contains 1
            # Find indices where bernoulli == 1
            flip_candidates = (bernoulli_samples[i] == 1).nonzero(as_tuple=True)[0]
            
            # Among candidates, pick the one with max flip rate
            # y^t ← C[indexOf(max(F_x[U == 1]))]
            max_idx = class_fliprate[i, flip_candidates].argmax()
            new_class = flip_candidates[max_idx]
            
            # Relabel
            y_train_flipped[i] = new_class
            total_flips += 1
    print(f"LNR: Total labels flipped: {total_flips} out of {len(y_train)} samples. {100*total_flips/len(y_train):.2f}%")
    return y_train_flipped


def evaluate_legacy(model, dataloader, criterion, device):
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
    return total_loss/len(dataloader.dataset), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, zero_division=0)



def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a dataset"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    model.train()
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(data_loader)
    
    return accuracy, avg_loss


def main(random_seed=42, dataset=None):
    # Config
    data_path = dataset
    model_path = "keel/mlp/best_mlp_chronic_disease_dataset.pth"
    
    # LNR Config
    # old t_flip
    #t_flip = 0.5
    lnr_epochs = 50  # Number of epochs to fine-tune last layer with LNR
    learning_rate = 0.001
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return 0,0,0,0
        
    X, y = parse_kaggle_dat(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_seed)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2. Prepare Base Model
    # We load the weights once here to pass them to CV and final training
    base_model = MLP(input_dim=X_train.shape[1])
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        base_state = torch.load(model_path, map_location=device)
        base_model.load_state_dict(base_state)
    else:
        print("ERROR: Pre-trained model not found!")
        return
    
    base_model = base_model.to(device)

    # 3. Generate Flipped Labels using LNR
    print("Generating flipped labels with LNR...")
    y_train = lnr(base_model,
        X_train,y_train,
        device,
        t_flip=0.5
    )

    
    # 4. Freeze Feature Layers (freeze all except the final classifier layer)
    # Freeze all parameters first
    for param in base_model.parameters():
        param.requires_grad = False

    # Unfreeze only the last Linear layer (assumes final module in `network` is the output Linear)
    for param in base_model.network[-1].parameters():
        param.requires_grad = True
    
    print("Trainable parameters:")
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")

    # 5. Prepare Data Loaders with Flipped Labels
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        y_train  # Already a tensor from lnr()
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 6. Setup Optimizer (only for trainable parameters)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, base_model.parameters()),
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    #Original result
    org_acc, org_loss = evaluate(base_model, test_loader, criterion, device)
    print(f"Original Test Accuracy (before LNR training): {org_acc:.2f}%")

    # 7. Train Only the Last Layer with Flipped Labels
    print(f"\nTraining classifier for {lnr_epochs} epochs with LNR labels...")
    base_model.train()
    
    for epoch in range(lnr_epochs):
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = base_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100. * correct / total
        
        # Evaluate on test set (with original labels)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            test_acc, test_loss = evaluate(base_model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{lnr_epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%")
    
    # 8. Final Evaluation
    print("\nFinal Evaluation on Test Set (Original Labels):")
    final_acc, final_loss = evaluate(base_model, test_loader, criterion, device)
    print(f"Test Accuracy: {final_acc:.2f}%")
    
    return final_acc
    

if __name__ == "__main__":
    # Ensure you run the original script first to generate 'best_mlp_glass0.pth'
    main(dataset="kaggle_data/chronic_disease_dataset.csv")