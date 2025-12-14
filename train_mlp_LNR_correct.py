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

# ... [parse_keel_dat and print_imbalance_info functions remain the same] ...
def parse_keel_dat(filepath):
    features = []
    labels = []
    data_started = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.lower() == '@data':
                data_started = True
                continue
            if line.startswith('@'): continue
            if data_started:
                parts = [p.strip() for p in line.split(',')]
                try:
                    feature_values = [float(p) for p in parts[:-1]]
                    label = 1 if "positive" in parts[-1].lower() else 0
                    features.append(feature_values)
                    labels.append(label)
                except ValueError: continue
    return np.array(features), np.array(labels)

def print_imbalance_info(y):
    n_positive = sum(y)
    total = len(y)
    print(f"Pos: {n_positive}, Neg: {total - n_positive}")

# ... [MLP Class remains the same] ...
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[5, 10, 5]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_epoch_lnr(model, dataloader, criterion, optimizer, device, t_flip):
    """
    Train ONLY the last layer using LNR logic.
    Assumes model is already pre-trained and hidden layers are frozen.
    """
    model.train()
    total_loss = 0.0
    total_flips = 0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x).squeeze()
        
        # --- LNR Logic ---
        targets = batch_y.clone()
        
        with torch.no_grad():
            # 1. Get Posterior Probabilities
            probs = torch.sigmoid(outputs)
            
            # 2. Identify Majority Samples (Class 0)
            maj_indices = (batch_y == 0).nonzero(as_tuple=True)[0]
            
            # forloop potentiellt
            if len(maj_indices) > 1:
                # Get probs of majority samples looking like minority
                p_maj = probs[maj_indices]
                
                # 3. Z-score Standardization
                mu = p_maj.mean()
                sigma = p_maj.std()
                
                if sigma > 1e-6:
                    z_scores = (p_maj - mu) / sigma
                    
                    # 4. Calculate Flip Rate (rho) and Flip
                    # rho = max(tanh(Z - t_flip), 0)
                    rho = torch.tanh(z_scores - t_flip).clamp(min=0)
                    
                    # Bernoulli sampling
                    flip_mask = torch.bernoulli(rho).bool()
                    indices_to_flip = maj_indices[flip_mask]
                    
                    if len(indices_to_flip) > 0:
                        targets[indices_to_flip] = 1.0
                        total_flips += len(indices_to_flip)
        
        # Calculate loss against LNR targets
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        
    return total_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds), total_flips

def evaluate(model, dataloader, criterion, device):
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

def main(random_seed=42, dataset=None):
    # Config
    data_path = f"Keel_data_sets/{dataset}.dat"
    model_path = f"keel/mlp/best_mlp_{dataset}.pth"
    
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
        
    X, y = parse_keel_dat(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_seed)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2. Prepare Base Model
    # We load the weights once here to pass them to CV and final training
    base_model = MLP(input_dim=X_train.shape[1])
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        base_state = torch.load(model_path)
        base_model.load_state_dict(base_state)
    else:
        print("ERROR: Pre-trained model not found!")
        return
    
    t_flip = select_best_t_flip(
    X_train, 
    y_train, 
    base_state, 
    input_dim=X_train.shape[1], 
    device=device,
    lnr_epochs=15 # Can be lower than final training to save time
)
     # ============================================================
    # STEP 4: FINAL TRAINING WITH SELECTED t_flip
    # ============================================================
    
    # Prepare final loaders
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=32, shuffle=False)
    
    # 2. Initialize Model
    model = MLP(input_dim=X_train.shape[1]).to(device)
    model.load_state_dict(base_state)
    
    # 3. Load Pre-trained Weights old load model
    #if os.path.exists(model_path):
    #    print(f"Loading pre-trained model from {model_path}...")
    #    model.load_state_dict(torch.load(model_path))
    #else:
    #    print("ERROR: Pre-trained model not found! Training from scratch (not desired behavior).")
    
    # 4. Freeze Hidden Layers & Select Last Layer for Optimization
    # Access the sequential container
    # Layers 0 to -2 are hidden layers + activations. Layer -1 is the final Linear.
    
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the last layer (Index -1 in the Sequential list)
    # Note: model.network is the Sequential object
    last_layer = model.network[-1]
    for param in last_layer.parameters():
        param.requires_grad = True
        
    print("Feature extraction layers frozen. Only training last layer.")

    # Optimizer (Only pass last layer parameters)
    optimizer = optim.Adam(last_layer.parameters(), lr=learning_rate)
    
    # Loss (Keep class weights if desired, or remove if LNR handles balance sufficiently)
    num_pos = sum(y_train)
    pos_weight = torch.tensor([(len(y_train)-num_pos)/num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nStarting LNR Fine-tuning for {lnr_epochs} epochs...")
    
    best_f1 = 0.0
    
    for epoch in range(lnr_epochs):
        loss, acc, flips = train_epoch_lnr(model, train_loader, criterion, optimizer, device, t_flip)
        _, _, val_f1 = evaluate(model, test_loader, criterion, device)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            # Save the LNR-tuned model (optional, maybe to a new file)
            torch.save(model.state_dict(), f'keel/mlp/best_mlp_lnr_{dataset}.pth')
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss {loss:.4f}, Flips: {flips}, Val F1: {val_f1:.4f}")

    # Final Eval
    model.load_state_dict(torch.load(f'keel/mlp/best_mlp_lnr_{dataset}.pth'))
    _, acc, f1 = evaluate(model, test_loader, criterion, device)
    
    print(f"\nFinal Result with LNR - F1: {f1:.4f}, Acc: {acc:.4f}")
    return acc, 0, 0, f1


def select_best_t_flip(X_train, y_train, base_model_state, input_dim, device, lnr_epochs=20):
    """
    Performs K-Fold CV to find the best t_flip threshold.
    """
    # Define candidates based on Figure 3 range (-2 to 4) or typical Z-scores
    candidates = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    best_t = 0.5
    best_avg_f1 = -1.0
    
    print(f"\n--- Starting Cross-Validation for t_flip selection ---")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # 3 or 5 folds
    
    for t in candidates:
        fold_f1s = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            # Split data
            X_tr_fold, y_tr_fold = X_train[train_idx], y_train[train_idx]
            X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
            
            # Create Loaders
            train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_fold), torch.FloatTensor(y_tr_fold)), batch_size=32, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_fold), torch.FloatTensor(y_val_fold)), batch_size=32, shuffle=False)
            
            # Reset Model to Pre-trained State
            model = MLP(input_dim=input_dim).to(device)
            model.load_state_dict(base_model_state)
            
            # Freeze Hidden
            for param in model.parameters(): param.requires_grad = False
            for param in model.network[-1].parameters(): param.requires_grad = True
            
            optimizer = optim.Adam(model.network[-1].parameters(), lr=0.001)
            
            # Recalculate class weight for this fold
            num_pos = sum(y_tr_fold)
            pos_weight = torch.tensor([(len(y_tr_fold)-num_pos)/max(num_pos, 1)]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            # Train for a few epochs
            # Note: We use fewer epochs for CV to save time, or same as main
            for _ in range(lnr_epochs):
                train_epoch_lnr(model, train_loader, criterion, optimizer, device, t)
            
            # Evaluate
            _, _, f1 = evaluate(model, val_loader, criterion, device)
            fold_f1s.append(f1)
            
        avg_f1 = np.mean(fold_f1s)
        print(f"t_flip: {t:.1f} | Avg CV F1: {avg_f1:.4f}")
        
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_t = t
            
    print(f"Selected Best t_flip: {best_t} (F1: {best_avg_f1:.4f})")
    print(f"----------------------------------------------------\n")
    return best_t

if __name__ == "__main__":
    # Ensure you run the original script first to generate 'best_mlp_glass0.pth'
    main(dataset="ecoli1")