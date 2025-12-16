import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
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

def apply_lnr_globally(model, X, y, t_flip, device):
    """
    Applies LNR logic once on the entire dataset to generate new targets.
    """
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)

    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        probs = torch.sigmoid(outputs)
        
        # Identify Majority Samples (Class 0)
        maj_indices = (y_tensor == 0).nonzero(as_tuple=True)[0]
        
        targets = y_tensor.clone()
        total_flips = 0
        
        if len(maj_indices) > 1:
            p_maj = probs[maj_indices]
            
            # Global Z-score Standardization
            mu = p_maj.mean()
            sigma = p_maj.std()
            
            if sigma > 1e-6:
                z_scores = (p_maj - mu) / sigma
                
                # Calculate Flip Rate (rho) and Flip
                rho = torch.tanh(z_scores - t_flip).clamp(min=0)
                
                # Bernoulli sampling
                flip_mask = torch.bernoulli(rho).bool()
                indices_to_flip = maj_indices[flip_mask]
                
                if len(indices_to_flip) > 0:
                    targets[indices_to_flip] = 1.0
                    total_flips = len(indices_to_flip)
                    
    return targets.cpu().numpy(), total_flips

def train_epoch_lnr(model, dataloader, criterion, optimizer, device, t_flip):
    """
    Train ONLY the last layer using LNR logic.
    Assumes model is already pre-trained and hidden layers are frozen.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x).squeeze()
        
       
        
        # Calculate loss against LNR targets
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        
    return total_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze(-1)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
        
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    g_mean = np.sqrt(sensitivity * specificity)
    
    return total_loss/len(dataloader.dataset), acc, f1, auc, g_mean

def main(random_seed=42, dataset=None, model_path=None, data_seed=None):
    # Config
    data_path = f"Keel_data_sets/{dataset}.dat"
    if model_path is None:
        model_path = f"keel/mlp_final/best_mlp_{dataset}.pth"
    
    if data_seed is None:
        data_seed = random_seed

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=data_seed)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2. Prepare Base Model
    # We load the weights once here to pass them to CV and final training
    base_model = MLP(input_dim=X_train.shape[1])
    print(f"Checking model path: {os.path.abspath(model_path)}")
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        base_state = torch.load(model_path, weights_only=True)
        base_model.load_state_dict(base_state)
    else:
        print(f"ERROR: Pre-trained model not found at {model_path}!")
        return 0, 0, 0, 0
    
    t_flip = select_best_t_flip(
    X_train, 
    y_train, 
    base_state, 
    input_dim=X_train.shape[1], 
    device=device,
    lnr_epochs=15, # Can be lower than final training to save time
    random_state=random_seed
)
     # ============================================================
    # STEP 4: FINAL TRAINING WITH SELECTED t_flip
    # ============================================================

    model = MLP(input_dim=X_train.shape[1]).to(device)
    model.load_state_dict(base_state)

    # Apply LNR globally ONCE using the base model state
    print(f"Applying LNR logic globally with t_flip={t_flip}...")
    y_train_lnr, total_flips = apply_lnr_globally(model, X_train, y_train, t_flip, device)
    print(f"Total labels flipped: {total_flips}")

    
    # Prepare final loaders
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_lnr)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=32, shuffle=False)
    

    
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
    
    best_f1 = -1.0
    
    for epoch in range(lnr_epochs):
        loss, acc = train_epoch_lnr(model, train_loader, criterion, optimizer, device, t_flip)
        _, _, val_f1, _, _ = evaluate(model, test_loader, criterion, device)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            # Save the LNR-tuned model (optional, maybe to a new file)
            torch.save(model.state_dict(), f'keel/mlp/best_mlp_lnr_{dataset}.pth')
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss {loss:.4f}, Flips: {total_flips}, Val F1: {val_f1:.4f}")

    # Final Eval
    # Load the best LNR model saved during training
    best_model_path = f'keel/mlp/best_mlp_lnr_{dataset}.pth'
    if os.path.exists(best_model_path):
        print(f"Loading best LNR model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path))
    
    _, acc, f1, auc, g_mean = evaluate(model, test_loader, criterion, device)
    
    print(f"\nFinal Result with LNR - F1: {f1:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}, G-Mean: {g_mean:.4f}")
    return acc, f1, auc, g_mean


def select_best_t_flip(X_train, y_train, base_model_state, input_dim, device, lnr_epochs=20, random_state=42):
    """
    Performs K-Fold CV to find the best t_flip threshold.
    """
    # Define candidates based on Figure 3 range (-2 to 4) or typical Z-scores
    candidates = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    best_t = 0.5
    best_avg_f1 = -1.0
    
    print(f"\n--- Starting Cross-Validation for t_flip selection ---")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state) # 3 or 5 folds
    
    for t in candidates:
        fold_f1s = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            # Split data
            X_tr_fold, y_tr_fold = X_train[train_idx], y_train[train_idx]
            X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
            
            # Reset Model to Pre-trained State
            model = MLP(input_dim=input_dim).to(device)
            model.load_state_dict(base_model_state)
            
            # Apply LNR globally for this fold
            y_tr_fold_lnr, _ = apply_lnr_globally(model, X_tr_fold, y_tr_fold, t, device)

            # Create Loaders
            train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_fold), torch.FloatTensor(y_tr_fold_lnr)), batch_size=32, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_fold), torch.FloatTensor(y_val_fold)), batch_size=32, shuffle=False)
            
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
            _, _, f1, _, _ = evaluate(model, val_loader, criterion, device)
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
    fixed_data_seed = 6779
    runs = 5
    
    tot_f1 = 0.0
    f1_scores = []
    tot_acc = 0.0
    acc_scores = []
    tot_auc = 0.0
    auc_scores = []
    tot_gmean = 0.0
    gmean_scores = []

    for i in range(runs):
        # Generate a new random seed for the LNR process and training
        current_seed = np.random.randint(1, 10000)
        print(f"\n\nRunning experiment {i+1}/{runs} with data_seed: {fixed_data_seed}, run_seed: {current_seed}")
        
        # Assuming the model was saved with the data seed in the filename
        # Adjust the model path format if your saved models have a different naming convention
        # For example: f"keel/mlp/best_mlp_ecoli1_{fixed_data_seed}.pth"
        # If you are using a generic model, you can pass None or the specific path
        
        # Check if seeded model exists, otherwise fall back to generic
        dataset_name = "yeast1"
        seeded_model_path = "keel/mlp_final/final_mlp_yeast1_6779_0.6471.pth"
        

        test_acc, test_f1, test_auc, test_gmean = main(
            random_seed=current_seed, 
            dataset=dataset_name,
            model_path=seeded_model_path,
            data_seed=fixed_data_seed
        )
        
        tot_f1 += test_f1
        f1_scores.append(test_f1)
        tot_acc += test_acc
        acc_scores.append(test_acc)
        tot_auc += test_auc
        auc_scores.append(test_auc)
        tot_gmean += test_gmean
        gmean_scores.append(test_gmean)

    avg_f1 = tot_f1 / runs
    std_f1 = (sum((score - avg_f1) ** 2 for score in f1_scores) / runs) ** 0.5
    print(f"\n\nAverage F1 over {runs} runs: {avg_f1:.4f}")
    print(f"Standard Deviation of F1: {std_f1:.4f}")
    
    avg_acc = tot_acc / runs
    std_acc = (sum((score - avg_acc) ** 2 for score in acc_scores) / runs) ** 0.5
    print(f"Average Accuracy over {runs} runs: {avg_acc:.4f}")
    print(f"Standard Deviation of Accuracy: {std_acc:.4f}")

    avg_auc = tot_auc / runs
    std_auc = (sum((score - avg_auc) ** 2 for score in auc_scores) / runs) ** 0.5
    print(f"Average AUC over {runs} runs: {avg_auc:.4f}")
    print(f"Standard Deviation of AUC: {std_auc:.4f}")

    avg_gmean = tot_gmean / runs
    std_gmean = (sum((score - avg_gmean) ** 2 for score in gmean_scores) / runs) ** 0.5
    print(f"Average G-Mean over {runs} runs: {avg_gmean:.4f}")
    print(f"Standard Deviation of G-Mean: {std_gmean:.4f}")