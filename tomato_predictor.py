import os
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd

# --- Constants ---
# These should match the training environment for the saved model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models/toato.pth'
SCALER_PATH = 'scalers/tomato_genetics_scaler.pkl'
PCA_PATH = 'scalers/tomato_genetics_pca.pkl'

# --- Model Definition (Copied from your training script) ---
# NOTE: Corrected __init__ from your script's _init_
class SelfAttention(nn.Module):
    """
    Optimized Self-Attention model matching the architecture in toato.pth.
    """
    def __init__(self, input_size, hidden_size, num_attention_heads, output_dim=1, dropout_prob=0.5):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"Hidden size ({hidden_size}) must be divisible by num heads ({num_attention_heads})")

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.feature_embedding = nn.Linear(1, hidden_size)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.out_dropout = nn.Dropout(dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, output_dim)
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        x = input_tensor.unsqueeze(-1)
        embedded_features = self.feature_embedding(x)
        mixed_query_layer = self.query(embedded_features)
        mixed_key_layer = self.key(embedded_features)
        mixed_value_layer = self.value(embedded_features)
        query_layer, key_layer, value_layer = self.transpose_for_scores(mixed_query_layer), self.transpose_for_scores(mixed_key_layer), self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / np.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        aggregated_context = context_layer.mean(dim=1)
        hidden = self.out_dropout(aggregated_context)
        hidden = self.LayerNorm(hidden + embedded_features.mean(dim=1))
        output = self.regressor(hidden)
        return output

# --- Prediction Function ---
def predict_tomato_yield(input_df):
    """
    Loads the tomato model and preprocessors, then returns predictions.
    Args:
        input_df (pd.DataFrame): DataFrame from the uploaded CSV.
    Returns:
        pd.DataFrame: A dataframe with predictions.
    """
    print("--- Starting Tomato Prediction ---")
    try:
        # 1. Load preprocessors (Scaler and PCA)
        scaler = joblib.load(SCALER_PATH)
        pca = joblib.load(PCA_PATH)
        print("✅ Tomato scaler and PCA loaded.")

        # 2. Load Model Architecture
        # The hyperparameters must match the trained model.
        # PCA reduces features to 100, so input_size is 100.
        # Other params are based on your training script's optuna search.
        model = SelfAttention(
            input_size=100, # From n_pca=100 in training
            hidden_size=256, # Example best param
            num_attention_heads=8, # Example best param
            dropout_prob=0.3 # Example best param
        ).to(DEVICE)
        
        # 3. Load trained weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"✅ Tomato model loaded onto {DEVICE}.")

        # 4. Preprocess the input data
        # The pipeline MUST match the training script: scale -> pca
        input_scaled = scaler.transform(input_df.values)
        input_reduced = pca.transform(input_scaled)
        input_tensor = torch.from_numpy(input_reduced).float().to(DEVICE)
        print("✅ Input data preprocessed successfully.")

        # 5. Make prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        predictions_df = pd.DataFrame(output.cpu().numpy(), columns=['Predicted_FW'], index=input_df.index)
        print("✅ Prediction complete.")
        return predictions_df

    except FileNotFoundError as e:
        print(f"ERROR: A required file was not found: {e}")
        # Re-raise the exception with a more user-friendly message
        raise FileNotFoundError(
            f"Could not find a required file: {e.filename}. "
            f"Please ensure '{MODEL_PATH}', '{SCALER_PATH}', and '{PCA_PATH}' all exist."
        ) from e
    except Exception as e:
        print(f"An unexpected error occurred during tomato prediction: {e}")
        raise e
