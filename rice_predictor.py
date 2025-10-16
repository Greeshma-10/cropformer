import os
import torch
import pandas as pd
import joblib
import torch.nn as nn
import numpy as np
from sklearn.impute import SimpleImputer

# ===================================================================
# 1. MODEL DEFINITION (Adapted from your Rice training script)
# ===================================================================
class SelfAttention(nn.Module):
    """
    Genetics-only model for Rice, combining CNN and self-attention layers.
    This architecture is adapted from your training script for inference.
    """
    def __init__(self, num_attention_heads, input_size, hidden_size, output_dim=1, kernel_size=3,
                 hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = torch.nn.Linear(input_size, self.all_head_size)
        self.key = torch.nn.Linear(input_size, self.all_head_size)
        self.value = torch.nn.Linear(input_size, self.all_head_size)
        self.attn_dropout = torch.nn.Dropout(attention_probs_dropout_prob)
        self.out_dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.dense = torch.nn.Linear(hidden_size, input_size)
        self.LayerNorm = torch.nn.LayerNorm(input_size, eps=1e-12)
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(input_size, output_dim)
        self.cnn = torch.nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)

    def forward(self, input_tensor):
        cnn_hidden = self.cnn(input_tensor.view(input_tensor.size(0), 1, -1))
        input_tensor_after_cnn = cnn_hidden.view(cnn_hidden.size(0), -1)

        mixed_query_layer = self.query(input_tensor_after_cnn)
        mixed_key_layer = self.key(input_tensor_after_cnn)
        mixed_value_layer = self.value(input_tensor_after_cnn)

        attention_scores = torch.matmul(mixed_query_layer, mixed_key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, mixed_value_layer)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor_after_cnn)
        output = self.out(self.relu(hidden_states))
        return output

# ===================================================================
# 2. PREDICTION FUNCTION
# ===================================================================
def predict_rice_yield(df_input):
    """
    Runs the full prediction pipeline for the rice model.
    This version assumes model files are already present locally.
    """
    print("--- Starting Rice Prediction ---")
    
    # --- Step 1: Load scaler and model ---
    try:
        scaler = joblib.load('scalers/rice_scaler.pkl')
        print("✅ Rice scaler loaded.")

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_size = scaler.n_features_in_
        
        # Check if the number of features in the input data matches the scaler.
        if df_input.shape[1] != input_size:
            error_msg = (
                f"Input data has the wrong number of features.\n"
                f" - The model was trained on {input_size} features.\n"
                f" - Your uploaded file has {df_input.shape[1]} features.\n"
                f"Please check your CSV file to ensure it has the correct number of columns."
            )
            raise ValueError(error_msg)

        model = SelfAttention(
            num_attention_heads=8,
            input_size=input_size,
            hidden_size=64, # Based on hidden_dim in your training script
            hidden_dropout_prob=0.5,
            attention_probs_dropout_prob=0.5
        ).to(DEVICE)
        
        model.load_state_dict(torch.load('models/rice.pth', map_location=DEVICE))
        model.eval()
        print("✅ Rice model loaded successfully.")

    except FileNotFoundError as e:
        raise RuntimeError(f"A required model file was not found: {e}. Ensure 'rice.pth' and 'rice_scaler.pkl' are in the correct folders.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the model or scalers: {e}")

    # --- Step 2: Preprocess input data ---
    try:
        data_values = df_input.values
        data_values[data_values == -9] = np.nan
        
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_imputed = imputer.fit_transform(data_values)

        X_scaled = scaler.transform(data_imputed)
        X_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)

    except Exception as e:
       raise ValueError(f"An error occurred during data preprocessing: {e}")
    
    # --- Step 3: Make Predictions ---
    with torch.no_grad():
        output = model(X_tensor)
    
    # Since there's no index column, we create a simple range index.
    predictions_df = pd.DataFrame(output.cpu().numpy(), columns=['Predicted_Yield'], index=pd.RangeIndex(start=1, stop=len(output)+1))
    
    print("✅ Rice prediction complete.")
    return predictions_df

