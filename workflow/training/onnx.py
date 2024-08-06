import os
from transformers import AutoTokenizer, WhisperProcessor
from optimum.onnxruntime import (
    ORTModelForQuestionAnswering, ORTModelForCausalLM,
    ORTModelForSequenceClassification, ORTModelForTokenClassification,
    ORTModelForSpeechSeq2Seq
)
from optimum.onnxruntime import ORTOptimizer
from huggingface_hub import HfApi

def convert_to_onnx(model_name, task, output_dir):
    print(f"Converting model: {model_name} for task: {task}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    task_mapping = {
        "embedding": (None, AutoTokenizer, "feature-extraction"),
        "ner": (ORTModelForTokenClassification, AutoTokenizer, None),
        "text_classification": (ORTModelForSequenceClassification, AutoTokenizer, None),
        "whisper_finetuning": (ORTModelForSpeechSeq2Seq, WhisperProcessor, None),
        "question_answering": (ORTModelForQuestionAnswering, AutoTokenizer, None),
        "causal_lm": (ORTModelForCausalLM, AutoTokenizer, None),
    }

    if task not in task_mapping:
        print(f"Task {task} is not supported for ONNX conversion in this script.")
        return None

    ORTModelClass, ProcessorClass, special_task = task_mapping[task]

    if task == "embedding":
        ort_optimizer = ORTOptimizer.from_pretrained(model_name)
        ort_optimizer.export(output_dir=output_dir, task=special_task)
    else:
        ort_model = ORTModelClass.from_pretrained(model_name, export=True)
        ort_model.save_pretrained(output_dir)

    processor = ProcessorClass.from_pretrained(model_name)
    processor.save_pretrained(output_dir)
    
    print(f"Conversion complete. Model saved to: {output_dir}")
    return output_dir

def push_onnx_to_hub(api, local_path, repo_name):
    api.create_repo(repo_id=repo_name, exist_ok=True)
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_name,
        repo_type="model"
    )
    print(f"ONNX model pushed to Hub: {repo_name}")
    return repo_name