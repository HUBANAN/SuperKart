
import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_Token")

api = HfApi(token=HF_Token)

# Ensure the 'deployment' folder exists and contains necessary files
deployment_folder_path = "/content/deployment"
if not os.path.exists(deployment_folder_path):
    print(f"Error: Deployment folder '{deployment_folder_path}' not found.")
    exit(1)

print("Uploading Gradio deployment to Hugging Face Space...")

api.upload_folder(
    folder_path=deployment_folder_path,
    repo_id="ANNELEARN/SuperKart",
    repo_type="space"
)

print("Gradio deployment uploaded successfully")
