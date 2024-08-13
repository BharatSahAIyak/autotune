import os
from transformers import AutoTokenizer, WhisperProcessor
from optimum.onnxruntime import (
    ORTModelForQuestionAnswering, ORTModelForCausalLM,
    ORTModelForSequenceClassification, ORTModelForTokenClassification,
    ORTModelForSpeechSeq2Seq
)
from optimum.onnxruntime import ORTOptimizer
import onnx
from onnxconverter_common import float16

def quantize_onnx_model(model_dir, quantization_type):
    for filename in os.listdir(model_dir):
        if filename.endswith('.onnx'):
            input_model_path = os.path.join(model_dir, filename)
            output_model_path = os.path.join(model_dir, f"quantized_{filename}")
            
            try:
                model = onnx.load(input_model_path)
                
                if quantization_type == "16-bit-float":
                    model_fp16 = float16.convert_float_to_float16(model)
                    onnx.save(model_fp16, output_model_path)
                elif quantization_type in ["8-bit", "16-bit-int"]:
                    from onnxruntime.quantization import quantize_dynamic, QuantType
                    
                    quant_type_mapping = {
                        "8-bit": QuantType.QInt8,
                        "16-bit-int": QuantType.QInt16,
                    }
                    weight_type = quant_type_mapping[quantization_type]
                    
                    quantize_dynamic(
                        model_input=input_model_path,
                        model_output=output_model_path,
                        weight_type=weight_type
                    )
                else:
                    print(f"Unsupported quantization type: {quantization_type}")
                    continue

                os.remove(input_model_path)
                os.rename(output_model_path, input_model_path)
                
                print(f"Quantized ONNX model saved to: {input_model_path}")
            except Exception as e:
                print(f"Error during ONNX quantization: {str(e)}")
                if os.path.exists(output_model_path):
                    os.remove(output_model_path)

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