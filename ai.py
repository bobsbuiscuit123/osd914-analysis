import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shap

# ==========================================
# 1. RESEARCH-GRADE MLP ARCHITECTURE
# ==========================================
class LiverBrainCrossTalkMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16, dropout_rate=0.3):
        super(LiverBrainCrossTalkMLP, self).__init__()
        
        # Shallow network architecture with Dropout to prevent overfitting on small n
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)


# ==========================================
# 2. MOCK DATA GENERATION (Simulating Neel's Output)
# ==========================================
def generate_mock_research_data():
    # Let's assume n=9 mouse samples from RR-8
    num_samples = 9 
    
    # Inputs: 12 filtered liver candidate miRNAs from Michelle's list
    num_liver_features = 12 
    
    # Outputs: 3 target brain pathway activation scores (e.g., NF-kB, Kynurenine, Redox)
    num_brain_targets = 3 
    
    np.random.seed(42)
    X_mock = np.random.randn(num_samples, num_liver_features).astype(np.float32)
    y_mock = np.random.randn(num_samples, num_brain_targets).astype(np.float32)
    
    return torch.tensor(X_mock), torch.tensor(y_mock)


# ==========================================
# 3. LEAVE-ONE-OUT CROSS-VALIDATION TRAINING LOOP
# ==========================================
def train_and_evaluate():
    X, y = generate_mock_research_data()
    num_samples, input_dim = X.shape
    output_dim = y.shape[1]
    
    print(f"Initialized Pipeline with {num_samples} samples.")
    print(f"Input features (Liver miRNAs): {input_dim} | Output targets (Brain Pathways): {output_dim}\n")
    
    # Leave-One-Out Cross-Validation is standard for small-n spaceflight omics
    oo_losses = []
    
    for i in range(num_samples):
        # Split into training (8 samples) and validation (1 sample)
        X_train = torch.cat([X[:i], X[i+1:]], dim=0)
        y_train = torch.cat([y[:i], y[i+1:]], dim=0)
        X_val, y_val = X[i:i+1], y[i:i+1]
        
        # Instantiate fresh model instance
        model = LiverBrainCrossTalkMLP(input_dim=input_dim, output_dim=output_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3) # L2 Regularization included
        
        # Training Loop
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            predictions = model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
            oo_losses.append(val_loss.item())
            
        print(f"LOOCV Fold {i+1}/{num_samples} - Validation MSE: {val_loss.item():.4f}")
        
    print(f"\nAverage System Cross-Talk Validation MSE: {np.mean(oo_losses):.4f}\n")
    return model, X


# ==========================================
# 4. SHAP EXPLAINABILITY INTEGRATION
# ==========================================
def compute_shap_explainability(model, X_tensor):
    print("--- Initiating SHAP Explainability Layer ---")
    model.eval()
    
    # Convert back to numpy for SHAP framework compatibility
    X_background = X_tensor.numpy()
    
    # Wrapper function for PyTorch to map numpy arrays through the torch model
    def model_predict_numpy(data_numpy):
        data_tensor = torch.tensor(data_numpy, dtype=torch.float32)
        with torch.no_grad():
            return model(data_tensor).numpy()
            
    # Initialize SHAP Explainer using KernelSHAP (ideal for model-agnostic omics ranking)
    explainer = shap.KernelExplainer(model_predict_numpy, X_background)
    shap_values = explainer.shap_values(X_background)
    
    print("SHAP values successfully computed.")
    print(f"Generated feature importance matrices for all {len(shap_values)} target pathways.")
    print("Ready to plot top driving liver hepatic anchors.")


# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    trained_model, final_data_tensor = train_and_evaluate()
    compute_shap_explainability(trained_model, final_data_tensor)