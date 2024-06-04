import logging
from celery import shared_task
from django.conf import settings
import pandas as pd
from .models import DatasetCreationTask
import os
from ragatouille import RAGTrainer
from django.conf import settings
from huggingface_hub import HfApi, login, snapshot_download


logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=settings.CELERY_MAX_RETRIES, retry_backoff=True)
def mine_negatives(self, request_data, user_id):
    try:
        task_id = self.request.id
        task = DatasetCreationTask.objects.get(id=task_id)
        dataset = request_data["dataset"]
        save_path = request_data["save_path"]
        model_checkpoint = request_data["model_checkpoint"]
        model_name = request_data["model_name"]
        hf_token = settings.HUGGING_FACE_TOKEN
        login(token=hf_token)
        output_data_path = "./mined_data/"
        task.status = "PROCESSING"
        task.save()

        ## Downloading dataset
        dataset_path = snapshot_download(dataset, repo_type="dataset")
        os.path.join(dataset_path, "dataset")
        chunks_df = pd.read_csv(
            os.path.join(dataset_path, "content.train.colbert.csv"),
        )[["c_id", "chunk"]]
        queries_df = pd.read_csv(
            os.path.join(dataset_path, "queries.train.colbert.csv"),
        )[["q_id", "question"]]
        mapping_df = pd.read_csv(
            os.path.join(dataset_path, "queries_row_mapping.train.colbert.csv")
        )[["q_id", "c_id"]]
        collection = chunks_df["chunk"].to_list()

        id_to_chunk = {}
        id_to_query = {}

        for _, row in chunks_df.iterrows():
            id_to_chunk[row["c_id"]] = row["chunk"]

        for _, row in queries_df.iterrows():
            id_to_query[row["q_id"]] = row["question"]

        pairs = []

        for _, row in mapping_df.iterrows():
            if row["c_id"] in id_to_chunk and row["q_id"] in id_to_query:
                pairs.append([id_to_query[row["q_id"]], id_to_chunk[row["c_id"]]])
            else:
                print("Chunk not found.")

        print("No of query-chunk: ", len(pairs))

        trainer = RAGTrainer(
            model_name=model_name,
            pretrained_model_name=model_checkpoint,
        )
        trainer.prepare_training_data(
            # TODO: Change here to all pairs
            raw_data=pairs[:100],
            data_out_path=output_data_path,
            all_documents=collection,
            num_new_negatives=10,
            mine_hard_negatives=True,
            positive_label="pos",
            negative_label="neg",
    )
        task.status = "UPLOADING"
        ## Upload to huggingface
        logger.info(f"Uploading mined data to {save_path}")
        api = HfApi(endpoint="https://huggingface.co", token=hf_token)
        api.create_repo(repo_id=save_path, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=output_data_path,
            path_in_repo="./",  # Upload to a specific folder
            repo_id=save_path,
            repo_type="dataset",
        )
        task.status = "COMPLETED"
        task.save()

    except Exception as e:
        logger.error(f"Error mining negatives for task {task_id}: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=5)


@shared_task(bind=True, max_retries=settings.CELERY_MAX_RETRIES, retry_backoff=True)
def generate_chunks(self, request_data, user_id):
    try:
        task_id = self.request.id
        task = DatasetCreationTask.objects.get(id=task_id)
        
        # file = request_data["file"]
        save_path = request_data["save_path"]
        # model_checkpoint = request_data["model_checkpoint"]
        # model_name = request_data["model_name"]
        # hf_token = settings.HUGGING_FACE_TOKEN
        # login(token=hf_token)
        output_data_path = "./mined_data/"
        task.status = "PROCESSING"
        task.save()
        
        

    except Exception as e:
        logger.error(f"Error mining negatives for task {task_id}: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=5)
