import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, WhisperProcessor, WhisperForConditionalGeneration
import os
import time
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, model_name, model_class, quantization_type, test_text=None):
        self.model_name = model_name
        self.model_class = model_class
        self.quantization_type = quantization_type
        self.test_text = test_text
        logger.info(f"Initializing ModelHandler with test_text: {test_text}")
        if model_class == WhisperForConditionalGeneration:
            self.processor = WhisperProcessor.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.original_model = self.load_model()
        self.quantized_model = self.load_quantized_model()

    def load_model(self):
        model = self.model_class.from_pretrained(self.model_name)
        model.to(self.device)
        return model
    
    def load_quantized_model(self):
        if self.quantization_type == "4-bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            model = self.model_class.from_pretrained(self.model_name, quantization_config=quantization_config)
        elif self.quantization_type == "8-bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = self.model_class.from_pretrained(self.model_name, quantization_config=quantization_config)
        elif self.quantization_type == "16-bit-static":
            model = self.model_class.from_pretrained(self.model_name)
            model = model.half()
        elif self.quantization_type == "16-bit-dynamic":
            model = self.model_class.from_pretrained(self.model_name)
            model = model.to(torch.float16)
        else:
            raise ValueError(f"Unsupported quantization type: {self.quantization_type}")
        
        if self.quantization_type not in ["4-bit", "8-bit"]:
            model.to(self.device)
        return model
    
    def run_inference(self, model, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()
        return outputs, end_time - start_time
    
    def get_model_size(self, model):
        torch.save(model.state_dict(), "temp.pth")
        size = os.path.getsize("temp.pth") / (1024 * 1024)
        os.remove("temp.pth")
        return size
    
    def compare_models(self):
        if self.test_text is None:
            logger.warning("No test text provided. Skipping inference testing.")
            return None, None

        logger.info(f"Running inference with test_text: {self.test_text}")
        original_outputs, original_time = self.run_inference(self.original_model, self.test_text)
        quantized_outputs, quantized_time = self.run_inference(self.quantized_model, self.test_text)
        
        original_size = self.get_model_size(self.original_model)
        quantized_size = self.get_model_size(self.quantized_model)
        
        logger.info(f"Original Model Size: {original_size:.2f} MB")
        logger.info(f"Quantized Model Size: {quantized_size:.2f} MB")
        logger.info(f"Original Model Inference Time: {original_time:.4f} seconds")
        logger.info(f"Quantized Model Inference Time: {quantized_time:.4f} seconds")
        
        return original_outputs, quantized_outputs
    
    def compare_outputs(self, original_outputs, quantized_outputs):
        if original_outputs is None or quantized_outputs is None:
            return

        if hasattr(original_outputs, 'logits') and hasattr(quantized_outputs, 'logits'):
            original_logits = original_outputs.logits.detach().cpu().numpy()
            quantized_logits = quantized_outputs.logits.detach().cpu().numpy()
            
            mse = ((original_logits - quantized_logits) ** 2).mean()
            spearman_corr, _ = spearmanr(original_logits.flatten(), quantized_logits.flatten())
            cosine_sim = cosine_similarity(original_logits.reshape(1, -1), quantized_logits.reshape(1, -1))[0][0]
            
            logger.info(f"\nMean Squared Error: {mse:.8f}")
            logger.info(f"Spearman Correlation: {spearman_corr:.8f}")
            logger.info(f"Cosine Similarity: {cosine_sim:.8f}")
        else:
            logger.info("Outputs do not have logits. Cannot compare.")

    def decode_output(self, outputs):
        raise NotImplementedError("This method should be implemented in subclasses")

    def quantize_and_compare(self):
        original_outputs, quantized_outputs = self.compare_models()
        if original_outputs is not None and quantized_outputs is not None:
            print("\nOriginal Model Output:")
            print(self.decode_output(original_outputs))
            print("\nQuantized Model Output:")
            print(self.decode_output(quantized_outputs))
            self.compare_outputs(original_outputs, quantized_outputs)
        return self.quantized_model