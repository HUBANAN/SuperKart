
import os

from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.getenv("HF_Token")

api = HfApi(token=HF_Token)

MODEL_REPO = "ANNELEARN/SuperKart-model"

create_repo(
repo_id=MODEL_REPO,
repo_type="model",
private=False,
exist_ok=True,
token=HF_Token
)

api.upload_file(
path_or_fileobj="best_superkart_model.pkl",
path_in_repo="best_superkart_model.pkl",
repo_id=MODEL_REPO,
repo_type="model"
)

print("Model uploaded successfully")
