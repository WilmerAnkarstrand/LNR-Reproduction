import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import copy
import matplotlib.pyplot as plt


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
    print(f"LNR: Q mean: {mean.cpu().numpy()}, std: {std.cpu().numpy()}")

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
            
            #I want to see which samples are being flipped from which class to which class in a plot
            #print(f"LNR: Sample {i} flipped from class {y_train[i].item()} to class {new_class.item()}")

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
    
    model.train()
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def main(random_seed=42, dataset=None, model_path = "keel/mlp/best_mlp_chronic_disease_dataset.pth", data_seed=None):
    # Config
    data_path = dataset
    
    if data_seed is None:
        data_seed = random_seed
    
    # LNR Config
    # old t_flip
    #t_flip = 0.5
    lnr_epochs = 200  # Number of epochs to fine-tune last layer with LNR
    learning_rate = 0.001
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return 0,0,0,0
        
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=data_seed)
    # Target percentages for the imbalanced dataset:
    # - Low classes: ~3.5% each, so ~7% total for both low classes
    # - Medium classes: ~17.5% each (so ~35% total, within 30-40% range)
    # - Majority (high) class: remaining ~58%
    # min_samples_per_class=15 ensures at least 10 samples in training after 70/30 split
    X_train, y_train = make_imbalanced_dataset(
        X_train, y_train, 
        high_classes=high_classes,
        medium_classes=medium_classes,
        low_classes=low_classes,
        medium_target_pct=0.175,  # Each medium class ~17.5% (total ~35%)
        low_target_pct=0.035,     # Each low class ~3.5% (total ~7%)
        min_samples_per_class=15,  # Ensures ~10+ in training set after split
        random_seed=data_seed
    )
    
    print("New Imbalanced Dataset Information:")
    print_imbalance_info(y_train)
    
    
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
        t_flip=1
    )

    print("LNR Dataset Information:")
    print_imbalance_info(y_train.cpu().numpy())
    
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
        lr=learning_rate, weight_decay=1e-4
    )

    # Calculate class weights to handle imbalance
    # Cast to int because labels are loaded as floats
    y_train_int = y_train.cpu().numpy().astype(int)
    class_counts = np.bincount(y_train_int)
    # Standard formula for balanced weights: n_samples / (n_classes * n_samples_j)
    num_classes_present = len(np.unique(y_train_int))
    total_samples = len(y_train_int)
    class_weights = total_samples / (num_classes_present * np.maximum(class_counts, 1))
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    #Original result
    org_loss, org_acc, _, _, _, org_preds, org_labels = evaluate(base_model, test_loader, criterion, device)
    print(f"Original Test Accuracy (before LNR training): {org_acc*100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(org_labels, org_preds, zero_division=0))

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
            test_loss, test_acc, _, _, _, _, _ = evaluate(base_model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{lnr_epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Test Acc: {test_acc*100:.2f}%")
    
    # 8. Final Evaluation
    print("\nFinal Evaluation on Test Set (Original Labels):")
    final_loss, final_acc, final_prec, final_rec, final_f1, preds, labels = evaluate(base_model, test_loader, criterion, device)
    print(f"Test Accuracy: {final_acc*100:.2f}%")

    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(labels, preds, zero_division=0))
    
    per_class_prec = precision_score(labels, preds, average=None, zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

    return final_acc, final_prec, final_rec, final_f1, per_class_prec, per_class_f1
    

if __name__ == "__main__":
    # Ensure you run the original script first to generate 'best_mlp_glass0.pth'
    fixed_data_seed = 5871
    runs = 5
    
    tot_f1 = 0.0
    f1_scores = []
    tot_prec = 0.0
    prec_scores = []
    
    all_per_class_prec = []
    all_per_class_f1 = []

    for i in range(runs):
        # Generate a new random seed for the LNR process and training
        current_seed = np.random.randint(1, 10000)
        print(f"\n\nRunning experiment {i+1}/{runs} with data_seed: {fixed_data_seed}, run_seed: {current_seed}")
        
        test_acc, test_prec, test_rec, test_f1, per_class_prec, per_class_f1 = main(
            random_seed=current_seed, 
            dataset="kaggle_data/chronic_disease_dataset.csv", 
            model_path=f"keel/mlp_final/final_mlp_chronic_disease_dataset_{fixed_data_seed}_{0.2546}.pth",
            data_seed=fixed_data_seed
        )
        
        tot_f1 += test_f1
        f1_scores.append(test_f1)
        tot_prec += test_prec
        prec_scores.append(test_prec)
        
        all_per_class_prec.append(per_class_prec)
        all_per_class_f1.append(per_class_f1)

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
    
      

     
