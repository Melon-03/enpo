import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from typing import Tuple, Optional


# 默认缓存目录
DEFAULT_CACHE_DIR = "/root/autodl-tmp/huggingface"


def load_pre_trained_llm(
    model_name: str, tokenizer_name: str, revision: str = "main", cache_dir: Optional[str] = None, **kwargs
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision, cache_dir=cache_dir, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_pre_trained_text_embedding_model(
    model_name: str, tokenizer_name: str, cache_dir: Optional[str] = None, **kwargs
) -> Tuple[AutoModel, AutoTokenizer]:
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    return model, tokenizer
