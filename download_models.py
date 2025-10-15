import gdown
import os

files = {
    "models/foxtailmillet_model.pth": "https://drive.google.com/file/d/11tybRa9nP8bvecjdqM7nZRKWKafxeZnL/view?usp=drive_link",
    "scalers/foxtailmillet_geno_scaler.pkl": "https://drive.google.com/file/d/1HFJP_t0O74Hkvb82dvFAU7J7ymh1U5I4/view?usp=sharing",
    "scalers/foxtailmillet_env_scaler.pkl": "https://drive.google.com/file/d/17W9u764a1-UNzp8FNkumwhhi8TNTx--a/view?usp=sharing",
    "attributions/top_SNP_attributions.csv": "https://drive.google.com/file/d/1xhuBNj06r93MLpjp7Lm6Fkb9ktTNRbUB/view?usp=sharing",
    "attributions/environment_attribution.csv": "https://drive.google.com/file/d/1_QXs9XEQlh-yOjGzqWIQc6li_KVD_Egp/view?usp=sharing",
}

def ensure_dir(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

for filepath, url in files.items():
    ensure_dir(filepath)
    if not os.path.exists(filepath):
        print(f"Downloading {filepath} ...")
        gdown.download(url, filepath, quiet=False)
    else:
        print(f"{filepath} already exists, skipping.")

print("Download complete.")
