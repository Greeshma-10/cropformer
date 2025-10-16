import os
import torch
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import torch.nn as nn
from lightning.pytorch import LightningModule

# --- Import Custom Predictor Modules ---
from tomato_predictor import predict_tomato_yield
from foxtail_millet_predictor import predict_foxtail_millet_yield
from rice_predictor import predict_rice_yield # <-- IMPORT RICE FUNCTION

# ===================================================================
# 1. SETUP AND CONFIGURATION
# ===================================================================

# --- Flask App Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model Hyperparameters (Must match your trained models) ---
MAIZE_INPUT_SIZE_GENO = 10003   # adjust to match the model’s saved state
WHEAT_INPUT_SIZE_GENO = 10000   # whatever it was trained with
WHEAT_INPUT_SIZE_ENV = 6        # since wheat uses 6 environmental variables
HIDDEN_SIZE = 64
BEST_PARAMS = {'num_attention_heads': 8, 'attention_probs_dropout_prob': 0.5}


# --- Device Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'--- Using device: {DEVICE} ---')

# ===================================================================
# 2. MODEL DEFINITIONS (for Maize and Wheat)
# ===================================================================

# --- Model 1: Genetics-Only Cropformer (for Maize) ---
class CropformerGeneticsOnly(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, output_dim=1, kernel_size=3,
                 hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5):
        super(CropformerGeneticsOnly, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(hidden_size, input_size)
        self.LayerNorm = nn.LayerNorm(input_size, eps=1e-12)
        self.relu = nn.ReLU()
        self.out = nn.Linear(input_size, output_dim)
        self.cnn = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)

    def forward(self, input_tensor):
        cnn_hidden = self.cnn(input_tensor.view(input_tensor.size(0), 1, -1))
        input_tensor_after_cnn = cnn_hidden
        mixed_query_layer = self.query(input_tensor_after_cnn)
        mixed_key_layer = self.key(input_tensor_after_cnn)
        mixed_value_layer = self.value(input_tensor_after_cnn)
        query_layer, key_layer, value_layer = mixed_query_layer, mixed_key_layer, mixed_value_layer
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor_after_cnn)
        output = self.out(self.relu(hidden_states.view(hidden_states.size(0), -1)))
        return output

# --- Model 2: GxE Cropformer (for Wheat) ---
class CropformerWithEnv(nn.Module):
    def __init__(self, geno_input_dim, env_input_dim, hidden_dim, num_attention_heads,
                 attention_probs_dropout_prob, hidden_dropout_prob=0.5, kernel_size=3):
        super(CropformerWithEnv, self).__init__()
        self.geno_cnn = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)
        self.geno_attention_head_size = int(hidden_dim / num_attention_heads)
        self.geno_query = nn.Linear(geno_input_dim, hidden_dim)
        self.geno_key = nn.Linear(geno_input_dim, hidden_dim)
        self.geno_value = nn.Linear(geno_input_dim, hidden_dim)
        self.geno_attn_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.geno_dense = nn.Linear(hidden_dim, geno_input_dim)
        self.geno_LayerNorm = nn.LayerNorm(geno_input_dim, eps=1e-12)
        self.env_branch = nn.Sequential(
            nn.Linear(env_input_dim, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 16)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(geno_input_dim + 16, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
        )
    def forward(self, x_geno, x_env):
        geno_cnn_hidden = self.geno_cnn(x_geno.view(x_geno.size(0), 1, -1))
        query = self.geno_query(geno_cnn_hidden)
        key = self.geno_key(geno_cnn_hidden)
        value = self.geno_value(geno_cnn_hidden)
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.geno_attention_head_size)
        probs = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(self.geno_attn_dropout(probs), value)
        geno_features = self.geno_LayerNorm(self.geno_dense(context) + geno_cnn_hidden).view(x_geno.size(0), -1)
        env_features = self.env_branch(x_env)
        combined_features = torch.cat((geno_features, env_features), dim=1)
        output = self.fusion_head(combined_features)
        return output

# ===================================================================
# 3. LOAD MODELS AND SCALERS (Done once when the app starts)
# ===================================================================
print("--- Loading models and scalers into memory... ---")
# --- Load Maize Model ---
try:
    maize_model = CropformerGeneticsOnly(
        **BEST_PARAMS, input_size=MAIZE_INPUT_SIZE_GENO, hidden_size=HIDDEN_SIZE
    ).to(DEVICE)
    maize_model.load_state_dict(torch.load('models/maize_model.pth', map_location=DEVICE))
    maize_model.eval()
    maize_scaler = joblib.load('scalers/maize_scaler.pkl')
    print("✅ Maize model and scaler loaded successfully.")
except Exception as e:
    maize_model = None
    print(f"⚠️ Could not load Maize model. Error: {e}")


# --- Load Wheat Model ---
try:
    wheat_model = CropformerWithEnv(
        geno_input_dim=WHEAT_INPUT_SIZE_GENO,
        env_input_dim=WHEAT_INPUT_SIZE_ENV,
        hidden_dim=HIDDEN_SIZE,
        **BEST_PARAMS
    ).to(DEVICE)
    wheat_model.load_state_dict(torch.load('models/wheat_model.pth', map_location=DEVICE))
    wheat_model.eval()
    wheat_geno_scaler = joblib.load('scalers/wheat_geno_scaler.pkl')
    wheat_env_scaler = joblib.load('scalers/wheat_env_scaler.pkl')
    print("✅ Wheat GxE model and scalers loaded successfully.")
except Exception as e:
    wheat_model = None
    print(f"⚠️ Could not load Wheat model. Error: {e}")

# Note: Tomato, Foxtail Millet, and Rice models are loaded on-demand.

# ===================================================================
# 4. FLASK ROUTES
# ===================================================================

def allowed_file(filename):
    """Checks if the uploaded file has a .csv extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    crop_type = request.form.get('crop_type')

    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    if file and crop_type:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # MODIFICATION: Special handling for rice CSV which has no header or index
            if crop_type == 'rice':
                df = pd.read_csv(filepath, header=None, index_col=False)
            else:
                # Default for other crops which have an index column
                df = pd.read_csv(filepath, index_col=0)
        except Exception as e:
            return render_template('index.html', error=f"Error reading CSV: {e}")

        predictions_df = None
        model_info = {}
        
        # --- Call the correct model ---
        if crop_type == 'maize' and maize_model:
            try:
                X_scaled = maize_scaler.transform(df.values)
                X_tensor = torch.from_numpy(X_scaled).to(torch.float32).to(DEVICE)

                with torch.no_grad():
                    output = maize_model(X_tensor)
                
                predictions_df = pd.DataFrame(output.cpu().numpy(), columns=['Predicted_DTT'], index=df.index)

                model_info = {
                    "crop": "Maize",
                    "trait": "Days to Tasseling",
                    "correlation": 0.9441
                }

            except Exception as e:
                return render_template('index.html', error=f"Maize prediction failed: {e}")

        elif crop_type == 'wheat' and wheat_model:
            try:
                marker_cols = [col for col in df.columns if 'Marker_' in col]
                env_cols = [col for col in df.columns if 'env_' in col]

                X_geno_scaled = wheat_geno_scaler.transform(df[marker_cols])
                X_env_scaled = wheat_env_scaler.transform(df[env_cols])
                
                X_geno_tensor = torch.from_numpy(X_geno_scaled).to(torch.float32).to(DEVICE)
                X_env_tensor = torch.from_numpy(X_env_scaled).to(torch.float32).to(DEVICE)

                with torch.no_grad():
                    output = wheat_model(X_geno_tensor, X_env_tensor)

                predictions_df = pd.DataFrame(output.cpu().numpy(), columns=['Predicted_TKW'], index=df.index)

                model_info = {
                    "crop": "Wheat",
                    "trait": "Thousand Kernel Weight",
                    "correlation": 0.6617
                }
            except Exception as e:
                return render_template('index.html', error=f"Wheat prediction failed: {e}")

        elif crop_type == 'tomato':
            try:
                # Delegate all tomato prediction logic to the specialized module
                predictions_df = predict_tomato_yield(df)

                model_info = {
                    "crop": "Tomato",
                    "trait": "Yield (predicted as Fruit Weight in grams)",
                    "correlation": 0.4883
                }

            except Exception as e:
                return render_template('index.html', error=f"Tomato prediction failed: {e}")

        elif crop_type == 'foxtail_millet':
            try:
                # Delegate all foxtail millet logic to its specialized module
                predictions_df = predict_foxtail_millet_yield(df)

                model_info = {
                    "crop": "Foxtail Millet",
                    "trait": "Grain Yield",
                    "correlation": "0.8183"
                }

            except Exception as e:
                return render_template('index.html', error=f"Foxtail Millet prediction failed: {e}")
        
        elif crop_type == 'rice':
            try:
                # Delegate all rice prediction logic to its specialized module
                predictions_df = predict_rice_yield(df)

                model_info = {
                    "crop": "Rice",
                    "trait": "Grain Yield",
                    "correlation": 0.4168
                }

            except Exception as e:
                return render_template('index.html', error=f"Rice prediction failed: {e}")
        
        # --- After a prediction is made, generate interpretations and summaries ---
        if predictions_df is not None and not predictions_df.empty:
            trait_interpretation = ""
            result_summary = ""

            if crop_type == 'maize':
                trait_interpretation = "Lower Days to Tasselling (DTT) indicates earlier flowering and potentially shorter crop duration."
                best_sample_row = predictions_df.loc[predictions_df['Predicted_DTT'].idxmin()]
                best_sample_id = best_sample_row.name
                best_value = best_sample_row['Predicted_DTT']
                result_summary = f"The standout sample is '{best_sample_id}' with the lowest Days to Tasseling ({best_value:.2f} days). This suggests it is the fastest-flowering variety, which could be ideal for shorter growing seasons."

            elif crop_type == 'wheat':
                trait_interpretation = "Higher Thousand Kernel Weight (TKW) generally correlates with better grain filling and yield."
                best_sample_row = predictions_df.loc[predictions_df['Predicted_TKW'].idxmax()]
                best_sample_id = best_sample_row.name
                best_value = best_sample_row['Predicted_TKW']
                result_summary = f"The standout sample is '{best_sample_id}' with the highest Thousand Kernel Weight ({best_value:.2f} grams). This indicates it has the best potential for high yield among the tested varieties."

            elif crop_type == 'tomato':
                trait_interpretation = "Higher Fruit Weight (FW) is a key component of and generally leads to increased overall yield."
                best_sample_row = predictions_df.loc[predictions_df['Predicted_FW'].idxmax()]
                best_sample_id = best_sample_row.name
                best_value = best_sample_row['Predicted_FW']
                result_summary = f"The standout sample is '{best_sample_id}' with the highest predicted Fruit Weight ({best_value:.2f} grams). This variety shows the greatest potential for producing large fruits and, consequently, a high yield."

            elif crop_type == 'foxtail_millet':
                trait_interpretation = "Higher Grain Yield indicates better overall productivity and performance for the given environmental conditions."
                # Find the best sample (highest Yield)
                best_sample_row = predictions_df.loc[predictions_df['Predicted_Yield'].idxmax()]
                best_sample_id = best_sample_row.name
                best_value = best_sample_row['Predicted_Yield']
                result_summary = f"The standout sample is '{best_sample_id}' with the highest predicted Grain Yield ({best_value:.2f}). This variety shows the most promise for high productivity in the specified environment."

            elif crop_type == 'rice':
                trait_interpretation = "Higher Grain Yield is desirable as it indicates better overall productivity for the rice variety."
                # Find the best sample (highest Yield)
                best_sample_row = predictions_df.loc[predictions_df['Predicted_Yield'].idxmax()]
                best_sample_id = best_sample_row.name
                best_value = best_sample_row['Predicted_Yield']
                result_summary = f"The standout sample is row '{best_sample_id}' with the highest predicted Grain Yield ({best_value:.2f}). This variety shows the greatest potential for high productivity."

            # --- Render results for ANY successful prediction ---
            return render_template(
                'index.html',
                predictions=predictions_df.round(2),
                model_info=model_info,
                interpretation=trait_interpretation,
                result_summary=result_summary
            )
        else:
            # Handle cases where prediction didn't run or returned empty
            return render_template('index.html',
                                   error=f"Model for '{crop_type}' not available or an error occurred during prediction.")
            
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)

