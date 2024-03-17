from django.conf import settings
from huggingface_hub import HfApi, HfFolder, Repository
import pandas as pd
from datetime import datetime
import os


def upload_dataset_to_hf(combined_data, dataset_name):
    huggingface_token = settings.HUGGING_FACE_TOKEN
    df = pd.DataFrame(combined_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_name = f"{dataset_name}_{timestamp}"

    csv_path = f"/tmp/{repo_name}.csv"
    df.to_csv(csv_path, index=False)

    hf_api = HfApi()

    repo_url = hf_api.create_repo(token=huggingface_token,
                                  name=repo_name,
                                  organization=None,
                                  repo_type="dataset",
                                  private=False)

    repo = Repository(local_dir=f"/tmp/{repo_name}", clone_from=repo_url, use_auth_token=huggingface_token)

    os.rename(csv_path, f"/tmp/{repo_name}/{repo_name}.csv")
    repo.git_add()
    repo.git_commit("Add dataset")

    repo.git_push()

    return repo_name
