import torch
import structlog
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.core.interfaces import BaseLLM
from typing import Any

logger = structlog.get_logger(__name__)

class HFClient(BaseLLM):
    """
    HuggingFace LLM client configured for Kaggle (Tesla T4/P100).
    Uses bitsandbytes for 4-bit quantization.
    """
    def __init__(
        self, 
        model: str = "Qwen/Qwen3-8B", 
        max_tokens: int = 512, 
        temperature: float = 0.7, 
        top_p: float = 0.9, 
        **kwargs: Any
    ):
        self._model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        logger.info("loading_hf_model", model=model, quantization="4-bit")
        
        # Kaggle GPU optimization (4-bit quantization)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        max_new_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        
        do_sample = temperature > 0.0
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode only the generated text (ignoring prompt)
        input_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return response

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        # In a local notebook, sync is sufficient unless explicitly managing asyncio thread pools
        return self.generate(prompt, **kwargs)
