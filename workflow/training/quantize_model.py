from .quantize import ModelHandler
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoModel,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor
)
import torch
import time
import os
import shutil

class SequenceClassificationHandler(ModelHandler):
    def decode_output(self, outputs):
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        return f"Predicted class: {predicted_class}"

class QuestionAnsweringHandler(ModelHandler):
    def run_inference(self, model, text):
        parts = text.split('QUES')
        context = parts[0].strip()
        question = parts[1].strip()
        inputs = self.tokenizer(question, context, return_tensors='pt').to(self.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()
        return outputs, end_time - start_time

    def decode_output(self, outputs):
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        answer_start = torch.argmax(start_logits)
        answer_end = torch.argmax(end_logits) + 1
        input_ids = self.tokenizer.encode(self.test_text)
        answer = self.tokenizer.decode(input_ids[answer_start:answer_end])
        return f"Answer: {answer}"

class TokenClassificationHandler(ModelHandler):
    def decode_output(self, outputs):
        predictions = torch.argmax(outputs.logits, dim=2)
        tokens = self.tokenizer.tokenize(self.test_text)
        predicted_labels = [self.original_model.config.id2label[t.item()] for t in predictions[0]]
        
        result = []
        for token, label in zip(tokens, predicted_labels):
            result.append(f"{token:<15} {label}")
        
        return "Tokens and their labels:\n" + "\n".join(result)

class CausalLMHandler(ModelHandler):
    def run_inference(self, model, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        end_time = time.time()
        return outputs, end_time - start_time

    def decode_output(self, outputs):
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return f"Generated text: {generated_text}"

class EmbeddingModelHandler(ModelHandler):
    def run_inference(self, model, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()
        return outputs.last_hidden_state.mean(dim=1), end_time - start_time

    def decode_output(self, outputs):
        return f"Embedding shape: {outputs.shape}"

class WhisperHandler(ModelHandler):
    def compare_models(self):
        original_size = self.get_model_size(self.original_model)
        quantized_size = self.get_model_size(self.quantized_model)
        
        print(f"Original Model Size: {original_size:.2f} MB")
        print(f"Quantized Model Size: {quantized_size:.2f} MB")
        
        return None, None

    def decode_output(self, outputs):
        return "Whisper model quantized successfully"

def quantize_model(model_name, model_class, quantization_type, test_text=None):
    handler_map = {
        AutoModelForSequenceClassification: SequenceClassificationHandler,
        AutoModelForQuestionAnswering: QuestionAnsweringHandler,
        AutoModelForTokenClassification: TokenClassificationHandler,
        AutoModelForCausalLM: CausalLMHandler,
        AutoModel: EmbeddingModelHandler,
        WhisperForConditionalGeneration: WhisperHandler
    }

    handler_class = handler_map.get(model_class)
    if handler_class:
        handler = handler_class(model_name, model_class, quantization_type, test_text)
        quantized_model = handler.quantize_and_compare()
        
        temp_dir = f"temp_quantized_{model_name.replace('/', '_')}"
        os.makedirs(temp_dir, exist_ok=True)
        
        quantized_model.save_pretrained(temp_dir)
        
        if hasattr(handler, 'tokenizer'):
            handler.tokenizer.save_pretrained(temp_dir)
        elif hasattr(handler, 'processor'):
            handler.processor.save_pretrained(temp_dir)
        else:
            print("Could not save tokenizer or processor")
        
        return temp_dir
    else:
        print(f"Model {model_name} can't be quantized as it's not supported.")
        return None