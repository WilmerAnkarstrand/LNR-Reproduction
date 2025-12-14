"""
MLP Training Script with LNR (Label Noise Rebalance) for KEEL Datasets
Binary classification with label flipping for imbalanced data
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import pickle


def parse_keel_dat(filepath):
    """Parse KEEL .dat file format and extract features and labels."""
    features = []
    labels = []
    data_started = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Start reading data after @data tag
            if line.lower() == '@data':
                data_started = True
                continue
            
            # Skip header lines (attributes, relation, etc.)
            if line.startswith('@'):
                continue
            
            # Parse data lines
            if data_started:
                parts = [p.strip() for p in line.split(',')]
                attribute_count = len(parts) - 1  # Last part is the label
                try:
                    feature_values = [float(p) for p in parts[:attribute_count]]
                    label = 1 if "positive" in parts[attribute_count].lower() else 0
                    features.append(feature_values)
                    labels.append(label)
                except ValueError:
                    continue

    return np.array(features), np.array(labels)


def print_imbalance_info(y):
    """Print imbalance information for the dataset."""
    n_positive = sum(y)
    n_negative = len(y) - n_positive
    total = len(y)
    ratio = n_negative / n_positive if n_positive > 0 else float('inf')
    
    print("\n" + "="*50)
    print("Dataset Imbalance Information")
    print("="*50)
    print(f"Total samples:    {total}")
    print(f"Positive (minority): {n_positive} ({100*n_positive/total:.1f}%)")
    print(f"Negative (majority): {n_negative} ({100*n_negative/total:.1f}%)")
    print(f"Imbalance ratio:  1:{ratio:.2f}")
    print("="*50 + "\n")
    
    return ratio


class MLPBackbone(nn.Module):
    """Feature extractor (backbone) - frozen during LNR training."""
    
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
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]  # Output dimension of backbone
    
    def forward(self, x):
        return self.network(x)


class Classifier(nn.Module):
    """Final classification layer - trained during LNR."""
    
    def __init__(self, feat_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    """Combined MLP for evaluation (backbone + classifier)."""
    
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
    
    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


class MLPOriginal(nn.Module):
    """Original MLP architecture with 1 output (matches train_mlp.py for loading pretrained weights)."""
    
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
        
        # Original architecture has 1 output for BCEWithLogitsLoss
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_probs_for_lnr(self, x):
        """Convert single logit output to 2-class probabilities for LNR."""
        logit = self.network(x).squeeze(-1)
        prob_positive = torch.sigmoid(logit)
        prob_negative = 1 - prob_positive
        # Return as [P(class=0), P(class=1)] for each sample
        return torch.stack([prob_negative, prob_positive], dim=1)
    
    def get_backbone_state_dict(self):
        """Extract backbone weights (all layers except final) for transfer."""
        state_dict = {}
        # The network is Sequential with pairs of (Linear, ReLU) + final Linear
        # We want all except the last Linear layer
        for name, param in self.network.named_parameters():
            # network.6.weight/bias is the final layer for hidden_dims=[5,10,5]
            # We keep layers 0-5 (3 Linear + 3 ReLU, but ReLU has no params)
            layer_idx = int(name.split('.')[0])
            # For [5,10,5]: layers are 0(Linear),1(ReLU),2(Linear),3(ReLU),4(Linear),5(ReLU),6(Linear)
            # We want to exclude the last Linear (index 6)
            if layer_idx < len(list(self.network)) - 1:
                state_dict[name] = param.data.clone()
        return state_dict


class IndexedTensorDataset(TensorDataset):
    """TensorDataset that also returns the index of each sample."""
    
    def __getitem__(self, index):
        return (index,) + super().__getitem__(index)


def label_noise_rebalance(train_dataloader, model, device, thre=3.0, class_num=2, 
                          read=False, store=False, dataset_name='none', uid='default',
                          use_original_model=False):
    """
    LNR (Label Noise Rebalance) algorithm adapted for binary classification.
    
    This function implements Algorithm 1 from the LNR paper:
    1. Compute predictions (η̂) for all training samples
    2. Calculate statistics (μ, σ) for each class
    3. Calculate Z-scores and noise rate (ρ)
    4. Bernoulli sampling to decide which labels to flip
    
    Args:
        train_dataloader: DataLoader for training data (must return index, x, target)
        model: Pre-trained classifier model
        device: torch device
        thre: Threshold for Z-score (t_flip in paper)
        class_num: Number of classes (2 for binary)
        read: Whether to read cached noise info
        store: Whether to store noise info to file
        dataset_name: Name of dataset for caching
        uid: Unique identifier for caching
        use_original_model: If True, model is MLPOriginal and uses get_probs_for_lnr()
    
    Returns:
        noise_info: Dictionary containing 'noise_flag' with indices to flip
    """
    cache_file = f'uid_{uid}_noise_info_{dataset_name}.pkl'
    
    if not read and not store:
        return None
    
    if read:
        if os.path.exists(cache_file):
            print(f'Reading cached noise info from {cache_file}')
            with open(cache_file, 'rb') as f:
                noise_info = pickle.load(f)
            return noise_info
        else:
            return None

    print("Computing LNR label noise rebalancing...")
    model.eval()
    
    # Step 1: Collect predictions for all samples
    # Run multiple passes to get stable predictions
    pre_dict = {}
    with torch.no_grad():
        for epoch in range(2):  # 2 passes for stability
            for batch_idx, (index, x, target) in enumerate(train_dataloader):
                x = x.to(device)
                target = target.to(device)
                
                # Get model predictions (softmax/sigmoid probabilities)
                if use_original_model:
                    # MLPOriginal: convert single logit to 2-class probs
                    probs = model.get_probs_for_lnr(x)
                    pospre = probs.cpu().detach().numpy()
                else:
                    # MLP with 2 outputs: use softmax
                    out = model(x)
                    pospre = F.softmax(out, dim=1).cpu().detach().numpy()
                
                target_np = target.long().cpu().detach().numpy()
                
                for i in range(len(x)):
                    idx_key = str(index[i].item() if isinstance(index[i], torch.Tensor) else index[i])
                    if idx_key in pre_dict:
                        pre_dict[idx_key].append((pospre[i], target_np[i]))
                    else:
                        pre_dict[idx_key] = [(pospre[i], target_np[i])]

    # Average predictions over multiple passes
    targets = np.array([t[0][1] for t in pre_dict.values()])
    for k, v in pre_dict.items():
        pre_dict[k] = (np.mean(np.array([item[0] for item in v]), axis=0), v[0][1])
    preds = np.array([t[0] for t in pre_dict.values()])
    
    classes = np.array(list(range(class_num)))
    
    # Step 2: Calculate class priors (for weighting flips from majority to minority)
    cls, cls_cnt = np.unique(targets, return_counts=True)
    priors = []
    for c in classes:
        if c in cls:
            priors.append(cls_cnt[np.where(cls == c)[0]][0])
        else:
            priors.append(0)
    
    priors = np.array(priors)
    # Normalize priors: higher value = more minority (more likely to receive flipped labels)
    priors = 1 - (priors - np.min(priors)) / (np.max(priors) - np.min(priors) + 1e-8)
    
    # Step 3: Calculate mean and std for each class
    # For class k, compute stats of P(k) for samples NOT belonging to k
    preds_mean, preds_std = [], []
    comp = {k: {} for k in classes}  # Track flip statistics
    
    for k in classes:
        other_class_mask = np.where(targets != k)[0]
        if len(other_class_mask) > 1:
            preds_mean.append(np.mean(preds[other_class_mask, k]))
            preds_std.append(np.std(preds[other_class_mask, k]))
        else:
            preds_mean.append(0)
            preds_std.append(1)  # Avoid division by zero
    
    preds_mean = np.array(preds_mean)
    preds_std = np.array(preds_std)
    
    # Step 4: Calculate Z-scores and flip rates for each sample
    zscores = {}
    fliprate = {}
    noisecnt = 0
    noise_flag = {}
    label_cnt = {key: 0 for key in range(len(classes))}
    
    for key, value in pre_dict.items():
        pred_prob, c = value
        
        # Z-score: how much this sample looks like each class compared to baseline
        zscores[key] = (pred_prob - preds_mean) / (preds_std + 1e-8)
        
        # Prior weight: encourage flipping FROM majority TO minority
        # Only allow flipping to classes more minority than current class
        prior_weight = np.array([max(p - priors[c], 0) for p in priors])
        
        # Flip rate: tanh(Z - threshold) * prior_weight
        fliprate[key] = np.tanh(zscores[key] - thre) * prior_weight
        
        # Step 5: Bernoulli sampling - flip if fliprate > random uniform
        uniform_rand = np.random.rand(len(classes))
        noise_class = np.where(fliprate[key] > uniform_rand)[0]
        
        if len(noise_class):
            # If multiple classes qualify, pick the one with highest flip rate
            compare_flip = fliprate[key][noise_class]
            highest_flip = np.where(compare_flip == max(compare_flip))[0]
            noise_idx = noise_class[highest_flip]
            
            noisecnt += len(noise_idx)
            noise_flag[key] = classes[noise_idx]
            label_cnt[classes[noise_idx][0]] += 1
            
            # Track flip statistics
            if classes[noise_idx][0] not in comp[c]:
                comp[c][classes[noise_idx][0]] = 1
            else:
                comp[c][classes[noise_idx][0]] += 1
        else:
            label_cnt[c] += 1
    
    noise_info = {'noise_flag': noise_flag}
    
    if store:
        with open(cache_file, 'wb') as file:
            pickle.dump(noise_info, file)
        print(f'Saved noise info to {cache_file}')
    
    print(f'Label flipping dict: {comp}')
    print(f'Total label flips: {noisecnt}')
    print(f'After rebalancing: {label_cnt}')
    
    return noise_info


def train_epoch_lnr(backbone, classifier, dataloader, criterion, optimizer, device, noise_info=None):
    """
    Train classifier for one epoch with LNR label flipping.
    Backbone is frozen (eval mode), only classifier is trained.
    """
    # Freeze backbone, train only classifier (like original LNR paper)
    backbone.eval()
    classifier.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    flip_count = 0
    
    for batch_idx, (index, batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).long()
        
        # Apply LNR label flipping
        if noise_info is not None:
            for j in range(len(index)):
                idx_key = str(index[j].item() if isinstance(index[j], torch.Tensor) else index[j])
                if idx_key in noise_info['noise_flag']:
                    flip_count += 1
                    batch_y[j] = noise_info['noise_flag'][idx_key][0]
        
        optimizer.zero_grad()
        
        # Forward pass: backbone features are detached (no gradient)
        with torch.no_grad():
            feat = backbone(batch_x)
        
        # Only classifier gets gradients
        outputs = classifier(feat.detach())
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, flip_count


def evaluate(backbone, classifier, dataloader, criterion, device):
    """Evaluate model on dataset."""
    backbone.eval()
    classifier.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # Handle both indexed and non-indexed dataloaders
            if len(data) == 3:
                _, batch_x, batch_y = data
            else:
                batch_x, batch_y = data
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).long()
            
            feat = backbone(batch_x)
            outputs = classifier(feat)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def main(random_seed=42, dataset=None, use_pretrained=True):
    """
    Main training function with LNR.
    
    Args:
        random_seed: Random seed for reproducibility
        dataset: Name of the KEEL dataset
        use_pretrained: Whether to use pretrained MLP for LNR calculation
    """
    # Configuration
    data_path = f"Keel_data_sets/{dataset}.dat"
    hidden_dims = [5, 10, 5]  # Hidden layer shape 5x10x5
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 200
    test_size = 0.3
    thre = 3.0  # LNR threshold (t_flip)
    
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and parse data
    print(f"\nLoading data from: {data_path}")
    X, y = parse_keel_dat(data_path)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Print imbalance information
    imb_ratio = print_imbalance_info(y)
    
    # Split data
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
    
    # Create indexed dataset for training (needed for LNR)
    train_dataset = IndexedTensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_for_lnr = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize backbone and classifier (separated for LNR training)
    input_dim = X_train.shape[1]
    backbone = MLPBackbone(input_dim=input_dim, hidden_dims=hidden_dims)
    backbone = backbone.to(device)
    classifier = Classifier(feat_dim=hidden_dims[-1], num_classes=2)
    classifier = classifier.to(device)
    
    print(f"\nBackbone architecture (frozen during LNR training):")
    print(backbone)
    print(f"\nClassifier architecture (trained during LNR):")
    print(classifier)
    
    # Load pretrained model for LNR calculation
    pretrained_path = f'keel/mlp/best_mlp_{dataset}.pth'
    lnr_model = None
    use_original_model = False
    
    if use_pretrained and os.path.exists(pretrained_path):
        print(f"\n{'='*60}")
        print(f"Loading pretrained model from: {pretrained_path}")
        print(f"{'='*60}")
        
        # Create model with original architecture (1 output) to load pretrained weights
        lnr_model = MLPOriginal(input_dim=input_dim, hidden_dims=hidden_dims)
        lnr_model = lnr_model.to(device)
        
        # Load pretrained weights
        pretrained_state = torch.load(pretrained_path, map_location=device)
        lnr_model.load_state_dict(pretrained_state)
        lnr_model.eval()
        use_original_model = True
        
        print("Successfully loaded pretrained model for LNR computation!")
        print(f"Pretrained model architecture:")
        print(lnr_model)
        
        # Transfer backbone weights from pretrained model to our backbone
        print("\nTransferring backbone weights from pretrained model...")
        pretrained_backbone_state = lnr_model.get_backbone_state_dict()
        backbone.network.load_state_dict(pretrained_backbone_state)
        print("Backbone weights transferred successfully!")
        
    else:
        if use_pretrained:
            print(f"\nWarning: No pretrained model found at {pretrained_path}")
        print("Will use warmup training to create model for LNR computation...")
        lnr_model = None
        use_original_model = False
    
    # Loss function with class weights
    num_positives = sum(y_train)
    num_negatives = len(y_train) - num_positives
    class_weights = torch.FloatTensor([1.0, num_negatives / num_positives]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer - ONLY trains the classifier, backbone is frozen
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )
    
    # Phase 1: Warmup training (only if no pretrained model)
    if lnr_model is None:
        print("\n" + "="*60)
        print("Phase 1: Warmup Training (no pretrained model available)")
        print("="*60)
        
        # For warmup, we train the full model (backbone + classifier)
        warmup_optimizer = optim.Adam(
            list(backbone.parameters()) + list(classifier.parameters()), 
            lr=learning_rate, weight_decay=1e-4
        )
        
        warmup_epochs = 20
        for epoch in range(warmup_epochs):
            backbone.train()
            classifier.train()
            for batch_idx, (index, batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).long()
                
                warmup_optimizer.zero_grad()
                feat = backbone(batch_x)
                outputs = classifier(feat)
                loss = criterion(outputs, batch_y)
                loss.backward()
                warmup_optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                val_loss, val_acc, _, _, val_f1, _, _ = evaluate(
                    backbone, classifier, test_loader, criterion, device
                )
                print(f"Warmup Epoch [{epoch+1:3d}/{warmup_epochs}] | Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Create a combined model for LNR computation
        class CombinedModelForLNR(nn.Module):
            def __init__(self, backbone, classifier):
                super().__init__()
                self.backbone = backbone
                self.classifier = classifier
            def forward(self, x):
                return self.classifier(self.backbone(x))
        
        lnr_model = CombinedModelForLNR(backbone, classifier)
        use_original_model = False
    else:
        print("\n" + "="*60)
        print("Phase 1: Skipped (using pretrained model for LNR)")
        print("="*60)
    
    # Phase 2: Compute LNR noise info using pretrained/warmed-up model
    print("\n" + "="*60)
    print("Phase 2: Computing LNR Label Noise Rebalancing")
    print("="*60)
    
    # Compute noise info (only once at the start, as per paper)
    noise_info = label_noise_rebalance(
        train_loader_for_lnr, 
        lnr_model, 
        device,
        thre=thre,
        class_num=2,
        read=False,
        store=True,
        dataset_name=dataset,
        uid=f'lnr_{random_seed}',
        use_original_model=use_original_model
    )
    
    # Phase 3: Training with LNR
    print("\n" + "="*60)
    print("Phase 3: Training with LNR Label Flipping (classifier only)")
    print("="*60)
    
    # Reinitialize only the classifier for fair comparison
    # Backbone keeps the pretrained/warmed-up weights
    classifier = Classifier(feat_dim=hidden_dims[-1], num_classes=2)
    classifier = classifier.to(device)
    
    # Optimizer - ONLY trains the classifier
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )
    
    best_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc, flip_count = train_epoch_lnr(
            backbone, classifier, train_loader, criterion, optimizer, device, noise_info
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
            backbone, classifier, test_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            # Save both backbone and classifier
            torch.save({
                'backbone': backbone.state_dict(),
                'classifier': classifier.state_dict()
            }, f'keel/mlp/best_mlp_lnr_{dataset}.pth')
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | "
                  f"Flips: {flip_count}")
    
    # Final evaluation
    print("\n" + "="*60)
    print(f"Training Complete! Best F1: {best_f1:.4f} at epoch {best_epoch}")
    print("="*60)
    
    checkpoint = torch.load(f'keel/mlp/best_mlp_lnr_{dataset}.pth')
    backbone.load_state_dict(checkpoint['backbone'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    test_loss, test_acc, test_prec, test_rec, test_f1, preds, labels = evaluate(
        backbone, classifier, test_loader, criterion, device
    )
    
    print("\nFinal Test Results (with LNR):")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
    
    return test_acc, test_prec, test_rec, test_f1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MLP Training with LNR')
    parser.add_argument('--dataset', type=str, default='glass0', help='KEEL dataset name')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs')
    parser.add_argument('--thre', type=float, default=3.0, help='LNR threshold')
    args = parser.parse_args()
    
    runs = args.runs
    tot_f1 = 0.0
    f1_scores = []
    
    for i in range(runs):
        random_seed = np.random.randint(1, 10000)
        print(f"\n\n{'='*70}")
        print(f"Run {i+1}/{runs} with random seed: {random_seed}")
        print(f"{'='*70}")
        
        test_acc, test_prec, test_rec, test_f1 = main(
            random_seed=random_seed, 
            dataset=args.dataset,
            use_pretrained=True
        )
        tot_f1 += test_f1
        f1_scores.append(test_f1)
    
    avg_f1 = tot_f1 / runs
    std_f1 = (sum((score - avg_f1) ** 2 for score in f1_scores) / runs) ** 0.5
    
    print(f"\n\n{'='*70}")
    print(f"Final Results over {runs} runs:")
    print(f"{'='*70}")
    print(f"Average F1: {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"All F1 scores: {[f'{s:.4f}' for s in f1_scores]}")
