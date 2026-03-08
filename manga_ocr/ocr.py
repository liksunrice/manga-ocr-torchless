import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, AutoTokenizer
from huggingface_hub import hf_hub_download
import os
from pathlib import Path
import jaconv
import re

class MangaOcr:
    def __init__(self, pretrained_model_name_or_path="mayocream/manga-ocr-onnx", force_cpu=False):
        print(f"Loading processor and tokenizer from {pretrained_model_name_or_path}...")
        self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        
        self.eos_token_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id
        
        model_path = Path(pretrained_model_name_or_path)
        encoder_path = model_path / "encoder_model.onnx"
        decoder_path = model_path / "decoder_model.onnx"

        if model_path.is_dir() and encoder_path.exists() and decoder_path.exists():
            print(f"Loading local ONNX models from {pretrained_model_name_or_path}...")
            encoder_path = str(encoder_path)
            decoder_path = str(decoder_path)
        else:
            print(f"Downloading ONNX files if not present from {pretrained_model_name_or_path}...")
            encoder_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="encoder_model.onnx")
            decoder_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="decoder_model.onnx")

        print("Loading ONNX Runtime sessions for MangaOCR...")
        # Prioritize hardware acceleration
        available_providers = ort.get_available_providers()
        preferred_providers = [
            'TensorrtExecutionProvider',   # NVIDIA High Performance
            'CUDAExecutionProvider',       # NVIDIA Standard
            'CoreMLExecutionProvider',     # Apple Silicon
            'DmlExecutionProvider',        # Windows DirectML
            'OpenVINOExecutionProvider',   # Intel
            'ROCmExecutionProvider',       # AMD Linux
            'CPUExecutionProvider'         # Fallback
        ]
        
        if force_cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = [p for p in preferred_providers if p in available_providers]
            if not providers:
                providers = ['CPUExecutionProvider']
            
        print(f"Using execution providers: {providers}")
        self.encoder_session = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder_session = ort.InferenceSession(decoder_path, providers=providers)
        # print("MangaOCR ONNX loaded successfully.")
        
    def __call__(self, img: Image.Image, max_length: int = 300) -> str:
        img = img.convert('L').convert('RGB')
        
        pixel_values = self.processor(img, return_tensors="np").pixel_values
        
        encoder_outputs = self.encoder_session.run(None, {"pixel_values": pixel_values})
        last_hidden_state = encoder_outputs[0]
        
        # Greedy Decoding
        input_ids = np.array([[self.bos_token_id]], dtype=np.int64)
        
        for _ in range(max_length):
            decoder_inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": last_hidden_state
            }
            
            try:
                logits = self.decoder_session.run(None, decoder_inputs)[0]
            except Exception as e:
                print(f"Decoder run failed: {e}")
                break
                
            next_token = np.argmax(logits[:, -1, :], axis=-1)[0]
            input_ids = np.concatenate([input_ids, np.array([[next_token]], dtype=np.int64)], axis=-1)
            
            if next_token == self.eos_token_id:
                break
        
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        text = ''.join(text.split())
        text = text.replace('…', '...')
        text = re.sub(r'[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
        text = jaconv.h2z(text, ascii=True, digit=True)
        
        return text
