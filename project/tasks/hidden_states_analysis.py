"""
分析模型各层hidden states，检查特定知识是否被遗忘
根据Reciprocal Rank来判断模型在每一层是否注意到特定关键词

对于指定的关键词，计算每一层hidden state通过lm_head得到的logits中，
该关键词token在所有vocab tokens中的倒数排名（Reciprocal Rank = 1/rank）
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
)

# 默认缓存目录，使用本地已下载的模型
DEFAULT_CACHE_DIR = "/root/autodl-tmp/huggingface"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """分析配置"""
    model_path: str
    model_type: str = "llama"
    base_model_path: str = None  # 基础模型路径（用于加载tokenizer和模型结构），如果模型路径是.pt文件则需要
    tokenizer_path: str = None  # tokenizer路径（可选，如果不指定则使用base_model_path或model_path）
    questions: List[str] = None
    keywords: List[str] = None
    output_keywords: List[str] = None  # 额外的关键词列表（与keywords合并，统一使用logits排名计算）
    max_new_tokens: int = 50
    temperature: float = 0.7
    do_sample: bool = False
    save_dir: str = "results/hidden_states_analysis"
    save_plot: bool = True
    plot_format: str = "png"
    save_data: bool = True

    def __post_init__(self):
        if self.questions is None:
            self.questions = ["Who wrote The Sun Dog?"]
        if self.keywords is None:
            self.keywords = ["stephen", "sun"]
        if self.output_keywords is None:
            self.output_keywords = []  # 额外的关键词列表，会与keywords合并统一处理


class HiddenStatesAnalyzer:
    """分析模型各层hidden states的类"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.hooks = []
        
    def load_model(self):
        """加载模型和tokenizer"""
        logger.info(f"Loading model from {self.config.model_path}...")
        
        # 判断是否是HuggingFace模型ID（包含斜杠但不是绝对路径，且不是本地存在的路径）
        is_hf_model_id = (
            '/' in self.config.model_path and 
            not os.path.isabs(self.config.model_path) and 
            not os.path.exists(self.config.model_path)
        )
        
        # 判断模型路径是否是.pt文件（checkpoint文件）
        is_checkpoint = self.config.model_path.endswith('.pt') or self.config.model_path.endswith('.pth')
        
        # 对于HuggingFace模型ID，跳过文件存在性检查
        if not is_hf_model_id and not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.config.model_path}")
        
        # 确定tokenizer路径
        if self.config.tokenizer_path:
            tokenizer_path = self.config.tokenizer_path
        elif self.config.base_model_path:
            tokenizer_path = self.config.base_model_path
        elif is_checkpoint:
            # 如果是checkpoint文件但没有指定base_model_path，尝试从同一目录查找
            model_dir = os.path.dirname(self.config.model_path)
            if os.path.exists(os.path.join(model_dir, "tokenizer.json")) or os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
                tokenizer_path = model_dir
            else:
                raise ValueError(
                    "Model path is a .pt checkpoint file. Please specify 'base_model_path' or 'tokenizer_path' "
                    "in the config file to load the tokenizer."
                )
        else:
            tokenizer_path = self.config.model_path
        
        # 加载tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                cache_dir=DEFAULT_CACHE_DIR
            )
        except Exception as e:
            logger.warning(f"Failed to load with AutoTokenizer: {e}")
            # 尝试使用本地文件
            if os.path.isdir(tokenizer_path):
                try:
                    # 尝试直接加载tokenizer文件
                    from transformers import PreTrainedTokenizer
                    self.tokenizer = PreTrainedTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                except:
                    raise RuntimeError(
                        f"Failed to load tokenizer from {tokenizer_path}. "
                        f"Please ensure the path contains tokenizer files or specify a valid 'base_model_path' or 'tokenizer_path'."
                    )
            else:
                raise RuntimeError(f"Failed to load tokenizer: {e}")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        if is_checkpoint:
            # 如果是checkpoint文件，需要先加载基础模型结构
            if not self.config.base_model_path:
                raise ValueError(
                    "Model path is a .pt checkpoint file. Please specify 'base_model_path' "
                    "in the config file to load the model structure."
                )
            
            logger.info(f"Loading base model structure from {self.config.base_model_path}...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    cache_dir=DEFAULT_CACHE_DIR
                )
                # 手动将模型移动到指定设备
                self.model = self.model.to(self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load base model from {self.config.base_model_path}: {e}")
            
            # 加载checkpoint权重
            logger.info(f"Loading checkpoint weights from {self.config.model_path}...")
            try:
                checkpoint = torch.load(self.config.model_path, map_location=self.device)
                # 处理不同的checkpoint格式
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # 加载权重
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("Checkpoint weights loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint weights: {e}. Using base model weights.")
        else:
            # 正常加载完整模型
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    cache_dir=DEFAULT_CACHE_DIR
                )
                # 手动将模型移动到指定设备
                self.model = self.model.to(self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")
        
        self.model.eval()
        
        # 获取层数
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            num_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            num_layers = len(self.model.transformer.h)
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            num_layers = len(self.model.gpt_neox.layers)
        else:
            num_layers = "unknown"
        
        logger.info(f"Model loaded successfully. Device: {self.device}")
        logger.info(f"Model has {num_layers} layers")
        
    def setup_hooks(self):
        """设置hooks来提取各层的hidden states"""
        self.hidden_states = {}
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # output通常是tuple，第一个元素是hidden states
                if isinstance(output, tuple):
                    hidden_state = output[0]
                else:
                    hidden_state = output
                # 保存hidden states [batch_size, seq_len, hidden_dim]
                # 确保移到CPU并detach
                if isinstance(hidden_state, torch.Tensor):
                    self.hidden_states[layer_idx] = hidden_state.detach().cpu()
                else:
                    # 如果不是tensor，尝试转换
                    self.hidden_states[layer_idx] = torch.tensor(hidden_state).detach().cpu()
            return hook
        
        # 为每一层注册hook
        layers = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            layers = self.model.transformer.layers
        else:
            # 尝试查找其他常见的层结构
            for attr in ['layers', 'h', 'blocks']:
                if hasattr(self.model, attr):
                    layers = getattr(self.model, attr)
                    break
            
            if layers is None:
                raise ValueError(
                    f"Cannot find model layers. Model structure: {type(self.model)}. "
                    f"Available attributes: {dir(self.model)}"
                )
        
        self.hooks = []
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(make_hook(i))
            self.hooks.append(hook)
        
        logger.info(f"Registered hooks for {len(self.hooks)} layers")
    
    def remove_hooks(self):
        """移除hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def tokenize_keywords(self, keywords: List[str]) -> Dict[str, List[int]]:
        """
        将关键词转换为token IDs
        
        注意：关键词可能以不同形式出现在问题中（大小写、标点等），
        这里我们保存原始关键词的token IDs，实际匹配时会进行灵活匹配
        """
        keyword_token_ids = {}
        for keyword in keywords:
            # 转换为小写进行tokenization（但保留原始关键词作为key）
            keyword_lower = keyword.lower().strip()
            tokens = self.tokenizer.encode(keyword_lower, add_special_tokens=False)
            keyword_token_ids[keyword] = tokens
            
            # 也尝试原始大小写
            if keyword != keyword_lower:
                tokens_original = self.tokenizer.encode(keyword, add_special_tokens=False)
                # 如果不同，保存两个版本（但使用原始keyword作为key）
                # 实际匹配时会尝试两种形式
                if tokens_original != tokens:
                    keyword_token_ids[f"{keyword}_original"] = tokens_original
        
        return keyword_token_ids
    
    def find_keyword_positions(
        self, 
        keyword_token_ids: List[int], 
        input_token_ids: List[int],
        keyword_text: str = None
    ) -> List[int]:
        """
        在输入序列中找到关键词token的位置
        
        Args:
            keyword_token_ids: 关键词的token IDs列表（可能包含多个子词）
            input_token_ids: 输入序列的token IDs
            keyword_text: 关键词的原始文本（用于文本级别匹配，大小写不敏感）
        
        Returns:
            关键词token在输入序列中的位置列表（返回所有匹配的位置）
        """
        positions = []
        
        if len(keyword_token_ids) == 0:
            return positions
        
        # 方法1：精确token ID匹配
        if len(keyword_token_ids) == 1:
            token_id = keyword_token_ids[0]
            for i, tid in enumerate(input_token_ids):
                if tid == token_id:
                    positions.append(i)
        else:
            # 如果关键词是多个token（子词），查找连续匹配
            for i in range(len(input_token_ids) - len(keyword_token_ids) + 1):
                if input_token_ids[i:i+len(keyword_token_ids)] == keyword_token_ids:
                    # 返回第一个token的位置（也可以返回所有位置）
                    positions.append(i)
                    # 也可以添加后续位置
                    for j in range(1, len(keyword_token_ids)):
                        if i + j < len(input_token_ids):
                            positions.append(i + j)
        
        # 方法2：如果精确匹配失败，尝试文本级别匹配（大小写不敏感）
        if len(positions) == 0 and keyword_text is not None:
            keyword_lower = keyword_text.lower().strip()
            # 获取关键词的文本形式（用于匹配）
            keyword_decoded = self.tokenizer.decode(keyword_token_ids).lower().strip()
            
            # 检查输入序列中每个token的文本形式
            for i, tid in enumerate(input_token_ids):
                token_text = self.tokenizer.decode([tid]).lower().strip()
                # 检查是否匹配（完全匹配或部分匹配）
                if keyword_lower == token_text or keyword_decoded == token_text:
                    positions.append(i)
                # 也检查是否包含关键词（用于处理子词情况）
                elif keyword_lower in token_text or token_text in keyword_lower:
                    # 如果关键词是token的一部分，也添加
                    if len(keyword_lower) >= 3:  # 只对较长的关键词进行部分匹配
                        positions.append(i)
        
        return positions
    
    def compute_output_token_reciprocal_rank(
        self,
        hidden_state: torch.Tensor,
        output_token_id: int,
        query_pos: int
    ) -> float:
        """
        计算输出token的Reciprocal Rank
        
        方法：使用hidden state通过lm_head计算logits，
        然后检查输出token在所有vocab tokens中的排名
        
        Args:
            hidden_state: [seq_len, hidden_dim] 某一层的hidden states
            output_token_id: 输出关键词的token ID
            query_pos: query token的位置
        
        Returns:
            Reciprocal Rank值 (0.0 到 1.0)
        """
        if query_pos >= len(hidden_state):
            return 0.0
        
        # 获取query位置的hidden state
        query_hidden = hidden_state[query_pos]  # [hidden_dim]
        
        # 获取lm_head层来计算logits
        if hasattr(self.model, 'lm_head'):
            lm_head = self.model.lm_head
        elif hasattr(self.model, 'embed_out'):
            lm_head = self.model.embed_out
        else:
            # 如果找不到lm_head，使用embedding层作为近似
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                embed_layer = self.model.model.embed_tokens
            else:
                return 0.0
            
            # 使用embedding相似度作为近似
            query_norm = query_hidden / (query_hidden.norm() + 1e-8)
            vocab_size = embed_layer.weight.shape[0]
            
            # 计算与所有vocab tokens的相似度（采样一部分以提高效率）
            # 对于大vocab，我们只检查输出token和随机采样的tokens
            output_embedding = embed_layer(torch.tensor([output_token_id]).to(self.device))[0]
            output_embedding = output_embedding.cpu()
            output_norm = output_embedding / (output_embedding.norm() + 1e-8)
            output_sim = torch.dot(query_norm, output_norm)
            
            # 采样一些随机tokens来估计排名
            sample_size = min(1000, vocab_size)
            sample_indices = torch.randint(0, vocab_size, (sample_size,))
            sample_embeddings = embed_layer(sample_indices.to(self.device)).cpu()
            sample_norms = sample_embeddings / (sample_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
            sample_sims = torch.matmul(sample_norms, query_norm)
            
            # 如果output_sim在sample中排名很高，估计rank=1
            if output_sim > sample_sims.max():
                return 1.0
            
            # 估算排名（简化处理）
            rank_estimate = (sample_sims > output_sim).sum().item() + 1
            if rank_estimate == 1:
                return 1.0
            else:
                # 由于是采样，返回一个估计值
                return 1.0 / rank_estimate if rank_estimate > 0 else 0.0
        
        # 使用lm_head计算logits
        query_hidden_gpu = query_hidden.to(self.device).unsqueeze(0)  # [1, hidden_dim]
        logits = lm_head(query_hidden_gpu)  # [1, vocab_size]
        logits = logits[0].cpu()  # [vocab_size]
        
        # 获取输出token的logit值
        output_logit = logits[output_token_id].item()
        
        # 计算该logit在所有vocab tokens中的排名
        # 由于vocab可能很大，我们使用argsort来找到排名
        sorted_indices = torch.argsort(logits, descending=True)
        rank = (sorted_indices == output_token_id).nonzero(as_tuple=True)[0].item() + 1
        
        # 计算Reciprocal Rank
        reciprocal_rank = 1.0 / rank if rank > 0 else 0.0
        
        return reciprocal_rank
    
    def analyze_question(self, question: str) -> Dict[str, List[float]]:
        """
        分析一个问题，返回每个关键词在每个层的Reciprocal Rank
        
        Returns:
            Dict[keyword, List[RR]] - 每个关键词在各层的Reciprocal Rank列表
        """
        logger.info(f"\nAnalyzing question: {question}")
        
        # 重置调试标志
        if hasattr(self, '_debug_printed'):
            delattr(self, '_debug_printed')
        
        # Tokenize输入
        inputs = self.tokenizer(question, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        input_tokens = input_ids[0].cpu().tolist()
        
        # 合并所有关键词（统一处理，不区分输入输出）
        all_keywords = list(self.config.keywords) + (list(self.config.output_keywords) if self.config.output_keywords else [])
        # Tokenize所有关键词
        all_keyword_token_ids = self.tokenize_keywords(all_keywords)
        
        # 调试：打印tokenization信息
        logger.info(f"  Input tokens: {input_tokens}")
        logger.info(f"  Input text tokens: {[self.tokenizer.decode([tid]) for tid in input_tokens]}")
        for keyword in all_keywords:
            tokens = all_keyword_token_ids.get(keyword, [])
            if len(tokens) == 0:
                keyword_lower = keyword.lower()
                tokens = all_keyword_token_ids.get(keyword_lower, [])
            logger.info(f"  Keyword '{keyword}' tokenized as: {tokens} -> {[self.tokenizer.decode([tid]) for tid in tokens]}")
        
        results = {keyword: [] for keyword in all_keywords}
        
        # 清空之前的hidden states
        self.hidden_states = {}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        # 获取各层的hidden states
        num_layers = len(self.hidden_states)
        
        # 获取query token的位置（通常是最后一个token，排除特殊token）
        # 找到最后一个非特殊token的位置
        query_end_idx = len(input_tokens) - 1
        # 跳过EOS token等特殊token，找到最后一个实际内容token
        for i in range(len(input_tokens) - 1, -1, -1):
            token_id = input_tokens[i]
            # 检查是否是特殊token（通常EOS token是特定ID）
            if token_id != self.tokenizer.eos_token_id and token_id != self.tokenizer.pad_token_id:
                query_end_idx = i
                break
        
        # 确保query_end_idx在有效范围内
        query_end_idx = min(query_end_idx, len(input_tokens) - 1)
        
        logger.info(f"  Query position: {query_end_idx}, Sequence length: {len(input_tokens)}")
        logger.info(f"  Query token: {self.tokenizer.decode([input_tokens[query_end_idx]])}")
        
        for layer_idx in range(num_layers):
            if layer_idx not in self.hidden_states:
                # 如果该层没有hidden states，跳过
                for keyword in all_keywords:
                    results[keyword].append(0.0)
                continue
            
            layer_hidden = self.hidden_states[layer_idx][0]  # [seq_len, hidden_dim]
            
            # 确保hidden states的长度与input tokens匹配
            if len(layer_hidden) != len(input_tokens):
                # 如果长度不匹配，可能需要截断或填充
                min_len = min(len(layer_hidden), len(input_tokens))
                layer_hidden = layer_hidden[:min_len]
                input_tokens_aligned = input_tokens[:min_len]
                query_end_idx_aligned = min(query_end_idx, min_len - 1)
            else:
                input_tokens_aligned = input_tokens
                query_end_idx_aligned = query_end_idx
            
            # 统一处理所有关键词：使用logits计算Reciprocal Rank
            for keyword in all_keywords:
                # 获取关键词的token IDs
                tokens = all_keyword_token_ids.get(keyword, [])
                
                # 如果原始形式没有匹配，尝试小写形式
                if len(tokens) == 0:
                    keyword_lower = keyword.lower()
                    tokens = all_keyword_token_ids.get(keyword_lower, [])
                
                if len(tokens) == 0:
                    results[keyword].append(0.0)
                    continue
                
                # 使用第一个token（如果是多token关键词，取第一个）
                keyword_token_id = tokens[0]
                
                # 计算该关键词token在logits中的Reciprocal Rank
                rr = self.compute_output_token_reciprocal_rank(
                    layer_hidden, keyword_token_id, query_end_idx_aligned
                )
                results[keyword].append(rr)
            
            # 每10层打印一次进度
            if (layer_idx + 1) % 10 == 0:
                logger.info(f"  Processed {layer_idx + 1}/{num_layers} layers...")
        
        return results
    
    def compute_reciprocal_rank_improved(
        self,
        hidden_states: torch.Tensor,
        keyword_token_ids: List[int],
        input_token_ids: List[int],
        query_pos: int,
        keyword_text: str = None
    ) -> float:
        """
        改进的Reciprocal Rank计算
        
        方法：计算query位置的hidden state与序列中所有位置的相似度，
        然后检查关键词token位置是否在top-1（即相似度最高）
        
        Args:
            hidden_states: [seq_len, hidden_dim] 某一层的hidden states
            keyword_token_ids: 关键词的token IDs列表
            input_token_ids: 输入序列的token IDs
            query_pos: query token的位置（通常是最后一个token）
            keyword_text: 关键词的原始文本（用于文本级别匹配）
        
        Returns:
            Reciprocal Rank值 (0.0 到 1.0)
        """
        if query_pos >= len(hidden_states) or len(hidden_states) == 0:
            return 0.0
        
        # 获取query位置的hidden state
        query_hidden = hidden_states[query_pos]  # [hidden_dim]
        
        # 计算query与所有位置的cosine相似度
        query_norm = query_hidden / (query_hidden.norm() + 1e-8)
        hidden_norms = hidden_states / (hidden_states.norm(dim=-1, keepdim=True) + 1e-8)
        similarities = torch.matmul(hidden_norms, query_norm)  # [seq_len]
        
        # 找到关键词token在输入序列中的位置（使用文本级别匹配作为后备）
        keyword_positions = self.find_keyword_positions(keyword_token_ids, input_token_ids, keyword_text=keyword_text)
        
        if len(keyword_positions) == 0:
            # 调试：只在第一层打印，避免日志过多
            if not hasattr(self, '_debug_printed'):
                logger.warning(f"    Keyword token IDs {keyword_token_ids} not found in input sequence")
                self._debug_printed = True
            return 0.0
        
        # 计算关键词位置的最大相似度
        keyword_similarities = [similarities[pos].item() for pos in keyword_positions if pos < len(similarities)]
        if len(keyword_similarities) == 0:
            return 0.0
        
        max_keyword_sim = max(keyword_similarities)
        
        # 计算该相似度在所有位置中的排名
        all_similarities = similarities.tolist()
        sorted_sims = sorted(all_similarities, reverse=True)
        
        # 找到max_keyword_sim的排名
        # 如果有多个相同值，取第一个出现的排名
        try:
            rank = sorted_sims.index(max_keyword_sim) + 1
            reciprocal_rank = 1.0 / rank
        except ValueError:
            # 如果找不到（不应该发生），返回0
            reciprocal_rank = 0.0
        
        return reciprocal_rank
    
    def plot_results(
        self, 
        all_results: Dict[str, Dict[str, List[float]]],
        question: str,
        save_path: Optional[str] = None
    ):
        """
        绘制Reciprocal Rank vs Layer的图表
        
        Args:
            all_results: Dict[question, Dict[keyword, List[RR]]]
            question: 当前问题
            save_path: 保存路径
        """
        results = all_results[question]
        num_layers = len(next(iter(results.values())))
        layers = list(range(num_layers))
        
        # 设置中文字体（如果可用）
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
        plt.figure(figsize=(12, 6))
        
        # 合并所有关键词（统一处理）
        all_keywords = list(self.config.keywords) + (list(self.config.output_keywords) if self.config.output_keywords else [])
        
        # 为每个关键词绘制一条线
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'v', 'D', 'p']
        for i, keyword in enumerate(all_keywords):
            if keyword not in results:
                continue
            rrs = results[keyword]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.plot(
                layers, rrs, 
                marker=marker, 
                label=keyword, 
                color=color, 
                linewidth=2, 
                markersize=6,
                markevery=max(1, num_layers // 20)  # 每隔几个点标记一次，避免过于密集
            )
        
        plt.xlabel('Layer', fontsize=14, fontweight='bold')
        plt.ylabel('Reciprocal Rank', fontsize=14, fontweight='bold')
        plt.title(f'Reciprocal Rank vs Layer\nQuestion: {question}', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim(-0.5, num_layers - 0.5)
        
        # 动态计算y轴范围：收集所有RR值
        all_rr_values = []
        for keyword in all_keywords:
            if keyword in results:
                all_rr_values.extend(results[keyword])
        
        if len(all_rr_values) > 0:
            min_rr = min(all_rr_values)
            max_rr = max(all_rr_values)
            
            # 添加一些边距（10%）
            if max_rr > min_rr:
                margin = (max_rr - min_rr) * 0.1
            elif max_rr > 0:
                margin = max_rr * 0.1
            else:
                margin = 0.01
            
            y_min = max(0, min_rr - margin)  # 确保最小值不小于0
            y_max = max_rr + margin
            
            plt.ylim(y_min, y_max)
            
            # 如果所有值都很小（小于0.01），使用科学计数法显示
            if max_rr < 0.01:
                # 设置y轴为科学计数法格式
                plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                plt.gca().ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
                plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
            else:
                # 设置合理的y轴刻度
                num_ticks = 6
                y_ticks = np.linspace(y_min, y_max, num_ticks)
                plt.yticks(y_ticks)
        else:
            # 如果没有数据，使用默认范围
            plt.ylim(-0.05, 1.05)
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # 设置x轴刻度
        if num_layers <= 24:
            plt.xticks(range(0, num_layers))
        else:
            step = max(1, num_layers // 12)
            plt.xticks(range(0, num_layers, step))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def get_model_save_dir(self) -> str:
        """
        根据模型路径生成保存目录名
        
        Returns:
            模型相关的目录名
        """
        model_path = self.config.model_path
        
        # 如果是.pt或.pth文件，使用"爷爷目录+父目录"命名
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            # 获取父目录和爷爷目录
            parent_dir = os.path.basename(os.path.dirname(model_path))
            grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
            
            # 组合爷爷目录和父目录
            if grandparent_dir and grandparent_dir not in ['saved-models', 'models', 'checkpoints', '']:
                if parent_dir and parent_dir not in ['saved-models', 'models', 'checkpoints', '']:
                    model_name = f"{grandparent_dir}_{parent_dir}"
                else:
                    model_name = grandparent_dir
            elif parent_dir and parent_dir not in ['saved-models', 'models', 'checkpoints', '']:
                model_name = parent_dir
            else:
                # 如果都不合适，使用文件名
                filename = os.path.splitext(os.path.basename(model_path))[0]
                model_name = filename if filename else "model"
        else:
            # 如果是目录或HuggingFace模型ID
            # 检查是否是HuggingFace模型ID（包含斜杠且不是绝对路径）
            if '/' in model_path and not os.path.isabs(model_path) and not os.path.exists(model_path):
                # 可能是HuggingFace模型ID，使用完整路径并替换斜杠为下划线
                model_name = model_path.replace('/', '_')
            else:
                # 如果是本地目录，使用目录名
                model_name = os.path.basename(model_path.rstrip('/'))
                if not model_name:
                    # 如果目录名为空，使用父目录名
                    model_name = os.path.basename(os.path.dirname(model_path))
        
        # 清理模型名称，只保留安全字符
        model_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_', '.')).strip()
        model_name = model_name.replace(' ', '_')
        
        # 如果模型名称太长，截断
        if len(model_name) > 50:
            model_name = model_name[:50]
        
        # 如果模型名称为空，使用默认名称
        if not model_name:
            model_name = "model"
        
        return model_name
    
    def run_analysis(self):
        """运行完整分析流程"""
        # 根据模型路径创建子目录
        model_save_dir = self.get_model_save_dir()
        full_save_dir = os.path.join(self.config.save_dir, model_save_dir)
        
        # 创建输出目录
        os.makedirs(full_save_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {full_save_dir}")
        
        # 加载模型
        self.load_model()
        
        # 设置hooks
        self.setup_hooks()
        
        try:
            # 分析每个问题
            all_results = {}
            for question in self.config.questions:
                results = self.analyze_question(question)
                all_results[question] = results
                
                # 绘制图表
                if self.config.save_plot:
                    try:
                        # 清理问题文本作为文件名
                        safe_question = "".join(c for c in question if c.isalnum() or c in (' ', '-', '_')).rstrip()
                        safe_question = safe_question.replace(' ', '_')[:50]
                        plot_path = os.path.join(
                            full_save_dir,
                            f"rr_vs_layer_{safe_question}.{self.config.plot_format}"
                        )
                        self.plot_results({question: results}, question, plot_path)
                    except Exception as e:
                        logger.warning(f"Failed to save plot for question '{question}': {e}")
                        logger.info("Continuing with data saving...")
            
            # 保存数据
            if self.config.save_data:
                data_path = os.path.join(full_save_dir, "analysis_results.json")
                # 转换为可序列化的格式
                serializable_results = {
                    q: {k: [float(rr) for rr in v] for k, v in r.items()}
                    for q, r in all_results.items()
                }
                with open(data_path, 'w') as f:
                    json.dump({
                        'model_path': self.config.model_path,
                        'questions': self.config.questions,
                        'keywords': self.config.keywords,
                        'output_keywords': self.config.output_keywords if self.config.output_keywords else [],
                        'results': serializable_results
                    }, f, indent=2)
                logger.info(f"\nData saved to {data_path}")
        
        finally:
            # 清理hooks
            self.remove_hooks()
        
        logger.info("\nAnalysis completed!")


def load_config(config_path: str) -> AnalysisConfig:
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 处理嵌套配置
    if 'output' in config_dict:
        output_config = config_dict.pop('output')
        config_dict.update({
            'save_dir': output_config.get('save_dir', 'results/hidden_states_analysis'),
            'save_plot': output_config.get('save_plot', True),
            'plot_format': output_config.get('plot_format', 'png'),
            'save_data': output_config.get('save_data', True),
        })
    
    if 'generation_config' in config_dict:
        gen_config = config_dict.pop('generation_config')
        config_dict.update({
            'max_new_tokens': gen_config.get('max_new_tokens', 50),
            'temperature': gen_config.get('temperature', 0.7),
            'do_sample': gen_config.get('do_sample', False),
        })
    
    # 确保output_keywords字段存在（如果没有，默认为["sorry"]）
    if 'output_keywords' not in config_dict:
        config_dict['output_keywords'] = ["sorry"]
    
    # 处理可选字段
    if 'base_model_path' not in config_dict:
        config_dict['base_model_path'] = None
    if 'tokenizer_path' not in config_dict:
        config_dict['tokenizer_path'] = None
    
    return AnalysisConfig(**config_dict)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="分析模型各层hidden states")
    parser.add_argument(
        "--config",
        type=str,
        default="config/hidden_states_analysis.yaml",
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建分析器并运行
    analyzer = HiddenStatesAnalyzer(config)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
