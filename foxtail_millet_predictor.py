import os
import torch
import pandas as pd
import joblib
import gdown
import torch.nn as nn
import torch.nn.functional as F

# ===================================================================
# 1. FILE DOWNLOADER
# ===================================================================

# Dictionary of files to download {destination_path: google_drive_id}
FILES_TO_DOWNLOAD = {
    "models/foxtailmillet_model.pth": "11tybRa9nP8bvecjdqM7nZRKWKafxeZnL",
    "scalers/foxtailmillet_geno_scaler.pkl": "1HFJP_t0O74Hkvb82dvFAU7J7ymh1U5I4",
    "scalers/foxtailmillet_env_scaler.pkl": "17W9u764a1-UNzp8FNkumwhhi8TNTx--a",
}

def download_files_if_needed():
    """Checks for the existence of model files and downloads them if missing."""
    print("--- Checking for Foxtail Millet model files... ---")
    for filepath, file_id in FILES_TO_DOWNLOAD.items():
        # Ensure the directory for the file exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if not os.path.exists(filepath):
            print(f"Downloading {filepath}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filepath, quiet=False)
        else:
            print(f"File '{filepath}' already exists. Skipping download.")
    print("--- File check complete. ---")


# ===================================================================
# 2. MODEL DEFINITION (Updated to match the new training script)
# ===================================================================
class Cropformer(nn.Module):
    """
    A Gene-by-Environment (GxE) interaction model for Foxtail Millet.
    This architecture is updated to match the final training script.
    """
    def __init__(self, geno_input_dim, env_input_dim, dropout=0.15):
        super(Cropformer, self).__init__()
        hidden_units_geno = 512
        hidden_units_env = 256
        
        self.snp_emb = nn.Linear(geno_input_dim, hidden_units_geno)
        self.env_fc = nn.Linear(env_input_dim, hidden_units_env)
        self.dropout = nn.Dropout(dropout)
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_units_geno + hidden_units_env, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x_geno, x_env):
        # Forward pass now matches the training script's logic
        snp_emb = self.dropout(F.relu(self.snp_emb(x_geno)))
        env_emb = self.dropout(F.relu(self.env_fc(x_env)))
        
        combined = torch.cat([snp_emb, env_emb], dim=1)
        
        return self.regressor(combined)

# ===================================================================
# 3. PREDICTION FUNCTION
# ===================================================================
def predict_foxtail_millet_yield(df_input):
    """
    Runs the full prediction pipeline for foxtail millet.
    """
    print("--- Starting Foxtail Millet GxE Prediction ---")
    
    # --- Step 1: Ensure model files are available ---
    download_files_if_needed()

    # --- Step 2: Load scalers and model ---
    try:
        geno_scaler = joblib.load('scalers/foxtailmillet_geno_scaler.pkl')
        env_scaler = joblib.load('scalers/foxtailmillet_env_scaler.pkl')
        print("✅ Foxtail Millet geno and env scalers loaded.")

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine model input dimensions from the scalers
        geno_input_dim = geno_scaler.n_features_in_
        env_input_dim = env_scaler.n_features_in_

        # Instantiate the new, corrected model class
        model = Cropformer(
            geno_input_dim=geno_input_dim,
            env_input_dim=env_input_dim,
            dropout=0.15 # Use the dropout from training
        ).to(DEVICE)
        
        model.load_state_dict(torch.load('models/foxtailmillet_model.pth', map_location=DEVICE))
        model.eval()
        print("✅ Foxtail Millet model loaded successfully.")

    except FileNotFoundError as e:
        raise RuntimeError(f"A required model file was not found: {e}. Please check the download paths.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the model or scalers: {e}")

    # --- Step 3: Preprocess input data ---
    try:
        # First, try to find columns using the standard prefixes
        marker_cols = [col for col in df_input.columns if 'Marker_' in col]
        env_cols = [col for col in df_input.columns if 'env_' in col]

        # **FALLBACK LOGIC**: If prefixes are not found, make an assumption
        if not marker_cols or not env_cols:
            print("⚠️ Warning: 'Marker_' or 'env_' prefixes not found in CSV.")
            print(f"    Assuming the last {env_input_dim} columns are environmental and the rest are genetic.")
            
            env_cols = df_input.columns[-env_input_dim:].tolist()
            marker_cols = df_input.columns[:-env_input_dim].tolist()
            
            if len(marker_cols) != geno_input_dim:
                # Improved, more descriptive error message
                error_msg = (
                    f"Input data has the wrong number of features.\n"
                    f"  - The model was trained on {geno_input_dim} genetic features.\n"
                    f"  - Your uploaded file has {len(marker_cols)} genetic features (after separating the last {env_input_dim} columns as environmental data).\n"
                    f"Please check your CSV file to ensure it has the correct number of columns."
                )
                raise ValueError(error_msg)

        # Separate the dataframe
        X_geno = df_input[marker_cols]
        X_env = df_input[env_cols]

        # Apply scaling
        X_geno_scaled = geno_scaler.transform(X_geno)
        X_env_scaled = env_scaler.transform(X_env)

        # Convert to tensors
        X_geno_tensor = torch.from_numpy(X_geno_scaled).float().to(DEVICE)
        X_env_tensor = torch.from_numpy(X_env_scaled).float().to(DEVICE)

    except Exception as e:
         raise ValueError(f"An error occurred during data preprocessing: {e}")
   
    # --- Step 4: Make Predictions ---
    with torch.no_grad():
        output = model(X_geno_tensor, X_env_tensor)
    
    predictions_df = pd.DataFrame(output.cpu().numpy(), columns=['Predicted_Yield'], index=df_input.index)
    
    print("✅ Foxtail Millet prediction complete.")
    return predictions_df

