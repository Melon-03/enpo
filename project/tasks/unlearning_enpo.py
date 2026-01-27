from omegaconf import DictConfig
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment
from project.eval import eval_llm
from hydra.utils import instantiate
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from typing import Literal, Optional
from project.utils.mean_pool import mean_pooling_reference_encoder
from project.utils import get_logger, log_hyperparameters
from lightning.pytorch.loggers import Logger
from project.utils.callbacks import get_default_callbacks
from project.utils.get_data_root import get_data_root
import os
import random
from copy import deepcopy

log = get_logger()


class UnlearningENPO:
    def __init__(
        self,
        global_config: DictConfig,
        target_id: str,
        target_name: str,
        pre_trained_llm: AutoModelForCausalLM,
        pre_trained_llm_tokenizer: AutoTokenizer,
        logger: Logger,
        **kwargs,
    ):
        self.global_config = global_config
        self.task_config = global_config.task
        self.target_id = target_id
        self.target_name = target_name
        self.pre_trained_llm = pre_trained_llm
        self.pre_trained_llm_tokenizer = pre_trained_llm_tokenizer
        self.logger = logger

    def unlearn(self):
        log.info("Task: unlearning_enpo")

        log.info("Validating config")
        for stage in self.task_config.stages:
            assert stage["type"] in ["training", "unlearning"], f"Invalid stage: {stage['type']}"
            assert stage["steps"] > 0, f"Steps must be greater than 0: {stage['steps']}"
            # Note: threshold is no longer required for eNPO loss, but kept for backward compatibility

        log.info("Instantiating text encoder")
        text_encoder, text_encoder_tokenizer = instantiate(self.task_config.text_encoder)

        # Load model from checkpoint if specified
        if hasattr(self.task_config, "model_checkpoint_path") and self.task_config.model_checkpoint_path is not None:
            log.info(f"Loading model from checkpoint: {self.task_config.model_checkpoint_path}")
            checkpoint = torch.load(self.task_config.model_checkpoint_path, map_location="cpu")
            state_dict = None
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and any(k.startswith("model.") or "pre_trained_llm" in k for k in checkpoint.keys()):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove "model." prefix if present (PyTorch Lightning format)
            if state_dict and any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
            
            self.pre_trained_llm.load_state_dict(state_dict, strict=False)
            log.info("Model loaded from checkpoint successfully")

        log.info("Instantiating embedding prediction model")
        log.info(
            f"Pre-trained model hidden size: {self.pre_trained_llm.config.hidden_size}"
        )
        log.info(f"Text encoder hidden size: {text_encoder.config.hidden_size}")
        embedding_prediction_model = instantiate(
            self.task_config.embedding_prediction_model,
            input_dim=self.pre_trained_llm.config.hidden_size,
            output_dim=text_encoder.config.hidden_size,
        )
        
        # Load embedding prediction model from checkpoint if specified
        if hasattr(self.task_config, "embedding_model_checkpoint_path") and self.task_config.embedding_model_checkpoint_path is not None:
            log.info(f"Loading embedding prediction model from checkpoint: {self.task_config.embedding_model_checkpoint_path}")
            checkpoint = torch.load(self.task_config.embedding_model_checkpoint_path, map_location="cpu")
            state_dict = None
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and any("embedding_prediction_model" in k for k in checkpoint.keys()):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove "model." or "embedding_prediction_model." prefix if present
            if state_dict and any(k.startswith("model.") or k.startswith("embedding_prediction_model.") for k in state_dict.keys()):
                state_dict = {k.replace("model.", "").replace("embedding_prediction_model.", ""): v 
                             for k, v in state_dict.items() 
                             if k.startswith("model.") or k.startswith("embedding_prediction_model.")}
            
            embedding_prediction_model.load_state_dict(state_dict, strict=False)
            log.info("Embedding prediction model loaded from checkpoint successfully")

        log_hyperparameters(
            self.logger,
            self.global_config,
            [
                ("pre_trained_llm", self.pre_trained_llm),
                ("text_encoder", text_encoder),
                ("embedding_prediction_model", embedding_prediction_model),
            ],
        )
        
        log.info("Loading training data")
        other_target_ids = []
        subfolders = [f for f in os.listdir(get_data_root()) if f != self.target_id]
        other_target_ids = subfolders[:self.task_config.num_other_targets]
        random.shuffle(other_target_ids)

        training_datamodule = instantiate(
            self.task_config.unlearning_data,
            primary_tokenizer=self.pre_trained_llm_tokenizer,
            secondary_tokenizer=text_encoder_tokenizer,
            target_ids=[self.target_id] + other_target_ids,
        )
        training_datamodule.prepare_data()
        training_datamodule.setup("train")

        log.info("Loading unlearning data")
        unlearning_datamodule = instantiate(
            self.task_config.unlearning_data,
            primary_tokenizer=self.pre_trained_llm_tokenizer,
            secondary_tokenizer=text_encoder_tokenizer,
            target_ids=[self.target_id],
        )
        unlearning_datamodule.prepare_data()
        unlearning_datamodule.setup("train")

        log.info("Instantiating UnlearningENPO")
        # Get reference model paths from config if specified
        reference_model_path = getattr(self.task_config, "reference_model_path", None)
        reference_embedding_model_path = getattr(self.task_config, "reference_embedding_model_path", None)
        
        task = UnlearningENPOTrainingModule(
            embedding_prediction_model=embedding_prediction_model,
            pre_trained_llm=self.pre_trained_llm,
            pre_trained_llm_tokenizer=self.pre_trained_llm_tokenizer,
            text_encoder=text_encoder,
            text_encoder_tokenizer=text_encoder_tokenizer,
            unlearning_target=self.target_name,
            reference_model_path=reference_model_path,
            reference_embedding_model_path=reference_embedding_model_path,
            **self.task_config.training_module,
        )

        log.info("Instantiating trainer")

        # 强制设置单机环境变量（防止SLURM检测）
        os.environ.pop("SLURM_JOB_ID", None)
        os.environ.pop("SLURM_NODEID", None)
        
        trainer = Trainer(
            **self.global_config.trainer,
            callbacks=get_default_callbacks(enable_checkpointing=False),  # 不保存checkpoint
            logger=self.logger,
            # plugins=[SLURMEnvironment(auto_requeue=False)],
            enable_checkpointing=False,
        )

        log.info("Starting initial evaluation!")
        if not self.global_config.skip_initial_eval:
            results = eval_llm(
                self.pre_trained_llm,
                self.pre_trained_llm_tokenizer,
                self.target_id,
                trainer.strategy.root_device,
                0,
            )
            trainer.logger.log_metrics(results)
        else:
            log.info("Skipping initial evaluation!")

        log.info("Starting training!")
        # 获取保存配置（默认为 True 以保持向后兼容）
        save_embedding_model = getattr(self.task_config, "save_embedding_model", True)
        save_unlearned_model = getattr(self.task_config, "save_unlearned_model", True)
        
        # 创建模型保存目录（仅在需要保存时创建）
        save_dir = None
        if save_embedding_model or save_unlearned_model:
            save_dir = Path("/root/autodl-tmp/saved-models") / self.target_id
            save_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"模型将保存到: {save_dir}")
        else:
            log.info("模型保存已禁用（根据配置）")
        
        # 找到最后一个training和unlearning阶段的索引
        last_training_idx = None
        last_unlearning_idx = None
        for idx, stage in enumerate(self.task_config.stages):
            if stage["type"] == "training":
                last_training_idx = idx
            elif stage["type"] == "unlearning":
                last_unlearning_idx = idx
        
        for idx, stage in enumerate(self.task_config.stages):
            log.info(
                f"Starting stage {idx + 1} ({stage['type']}) of {len(self.task_config.stages)}"
            )
            new_max_steps = (
                stage["steps"] if idx == 0 else stage["steps"] + trainer.max_steps
            )
            log.info(f"Setting max steps to {new_max_steps}")
            trainer.fit_loop.epoch_loop.max_steps = new_max_steps
            task.update_stage(stage["type"])
            if stage["type"] == "training":
                trainer.fit(task, datamodule=training_datamodule)
                # Clear GPU cache after training to free up memory for unlearning stage
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log.info("Cleared GPU cache after training stage")
                # 只在最后一个training阶段保存embedding_prediction_model（如果配置允许）
                if idx == last_training_idx and save_embedding_model:
                    embedding_model_path = save_dir / "embedding_prediction_model.pt"
                    log.info(f"保存embedding_prediction_model到: {embedding_model_path}")
                    # 获取原设备
                    original_device = next(task.embedding_prediction_model.parameters()).device
                    # 将模型移到CPU以节省GPU内存并保存
                    task.embedding_prediction_model = task.embedding_prediction_model.cpu()
                    torch.save(task.embedding_prediction_model.state_dict(), embedding_model_path)
                    # 恢复模型到原设备
                    task.embedding_prediction_model = task.embedding_prediction_model.to(original_device)
                    log.info(f"embedding_prediction_model已保存")
                elif idx == last_training_idx and not save_embedding_model:
                    log.info("跳过保存embedding_prediction_model（根据配置）")
            elif stage["type"] == "unlearning":
                # Note: threshold is no longer used in eNPO loss, but kept for backward compatibility
                if "threshold" in stage:
                    task.update_unlearning_threshold(stage["threshold"])
                trainer.fit(task, datamodule=unlearning_datamodule)
            else:
                raise ValueError(f"Invalid stage: {stage['type']}")
            log.info(f"Stage {idx + 1} ({stage['type']}) completed!")
            if stage["type"] == "unlearning":
                log.info("Starting testing!")
                results = eval_llm(
                    self.pre_trained_llm,
                    self.pre_trained_llm_tokenizer,
                    self.target_id,
                    device=trainer.strategy.root_device,
                    stage_number=idx + 1,
                )
                trainer.logger.log_metrics(results)
                # 只在最后一个unlearning阶段保存被遗忘的目标模型（pre_trained_llm）（如果配置允许）
                if idx == last_unlearning_idx and save_unlearned_model:
                    unlearned_model_path = save_dir / "unlearned_model.pt"
                    log.info(f"保存被遗忘的目标模型到: {unlearned_model_path}")
                    # 获取原设备
                    original_device = next(self.pre_trained_llm.parameters()).device
                    # 将模型移到CPU以节省GPU内存并保存
                    self.pre_trained_llm = self.pre_trained_llm.cpu()
                    torch.save(self.pre_trained_llm.state_dict(), unlearned_model_path)
                    # 恢复模型到原设备
                    self.pre_trained_llm = self.pre_trained_llm.to(original_device)
                    log.info(f"被遗忘的目标模型已保存")
                elif idx == last_unlearning_idx and not save_unlearned_model:
                    log.info("跳过保存被遗忘的目标模型（根据配置）")
        log.info("Unlearning complete!")


class UnlearningENPOTrainingModule(pl.LightningModule):
    def __init__(
        self,
        embedding_prediction_model: nn.Module,
        pre_trained_llm: AutoModelForCausalLM,
        pre_trained_llm_tokenizer: AutoTokenizer,
        text_encoder: AutoModel,
        text_encoder_tokenizer: AutoTokenizer,
        unlearning_target: str,
        training_warmup_steps: int,
        training_lr: float,
        training_weight_decay: float,
        unlearning_lr: float,
        unlearning_weight_decay: float,
        pretrained_model_hook_layer: int,
        clip_grad_norm: float,
        beta: float = 1.0,
        l1_weight: float = 1.0,
        l2_weight: float = 0.1,
        stage: Literal["training", "unlearning"] = "training",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=("embedding_prediction_model", "pre_trained_llm", "text_encoder")
        )
        self.unlearning_similarity_threshold = None
        self.embedding_prediction_model = embedding_prediction_model
        self.pre_trained_llm = pre_trained_llm
        self.pre_trained_llm_tokenizer = pre_trained_llm_tokenizer
        self.text_encoder = text_encoder
        self.text_encoder_tokenizer = text_encoder_tokenizer
        self.unlearning_target = unlearning_target
        self.beta = beta
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.stage = stage
        # Reference models for eNPO (will be set when switching to unlearning stage or loaded from path)
        self.embedding_prediction_model_ref = None
        self.pre_trained_llm_ref = None
        self.reference_model_path = kwargs.get("reference_model_path", None)
        self.reference_embedding_model_path = kwargs.get("reference_embedding_model_path", None)
        self.update_stage(stage)
        self.automatic_optimization = False

        log.info(f"Unlearning target: {unlearning_target}, computing embedding...")
        unlearning_target_tokens = text_encoder_tokenizer.encode(
            unlearning_target, add_special_tokens=True
        )
        unlearning_target_tokens = torch.tensor(
            unlearning_target_tokens, device=self.device
        ).unsqueeze(0)  # (1, seq_len)
        with torch.no_grad():
            self.unlearning_target_embedding = mean_pooling_reference_encoder(
                text_encoder(
                    input_ids=unlearning_target_tokens,
                    output_hidden_states=True,
                ),
                attention_mask=torch.ones_like(unlearning_target_tokens, device=self.device),
            )[0]
        # Detach to ensure it's not part of any computation graph
        self.unlearning_target_embedding = self.unlearning_target_embedding.detach()

    def on_fit_start(self):
        # Ensure unlearning_target_embedding is detached and on the right device
        self.unlearning_target_embedding = self.unlearning_target_embedding.detach().to(
            self.device
        )
        # Keep reference models on CPU to save GPU memory
        # They will be used on CPU in training_step with no_grad context
        # No need to move them to device here

    def _disable_grad(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    def _enable_grad(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = True

    def update_stage(self, stage: Literal["training", "unlearning"]):
        log.info(f"Updating stage to {stage}")
        # Save or load reference models when switching from training to unlearning
        if self.stage == "training" and stage == "unlearning":
            log.info("Setting up reference models for eNPO")
            
            # Get current device to restore models later
            current_device = next(self.embedding_prediction_model.parameters()).device
            
            # Load reference embedding_prediction_model from path if specified, otherwise deepcopy current
            if self.reference_embedding_model_path is not None:
                log.info(f"Loading reference embedding_prediction_model from: {self.reference_embedding_model_path}")
                checkpoint = torch.load(self.reference_embedding_model_path, map_location="cpu")
                # Temporarily move model to CPU for deepcopy to save memory
                self.embedding_prediction_model = self.embedding_prediction_model.cpu()
                torch.cuda.empty_cache()  # Clear GPU cache
                self.embedding_prediction_model_ref = deepcopy(self.embedding_prediction_model)
                # Restore model to original device
                self.embedding_prediction_model = self.embedding_prediction_model.to(current_device)
                
                state_dict = None
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif isinstance(checkpoint, dict) and any("embedding_prediction_model" in k for k in checkpoint.keys()):
                    state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Remove "model." or "embedding_prediction_model." prefix if present
                if state_dict and any(k.startswith("model.") or k.startswith("embedding_prediction_model.") for k in state_dict.keys()):
                    state_dict = {k.replace("model.", "").replace("embedding_prediction_model.", ""): v 
                                 for k, v in state_dict.items() 
                                 if k.startswith("model.") or k.startswith("embedding_prediction_model.")}
                
                self.embedding_prediction_model_ref.load_state_dict(state_dict, strict=False)
                log.info("Reference embedding_prediction_model loaded from checkpoint")
            else:
                log.info("Saving current embedding_prediction_model as reference")
                # Temporarily move model to CPU for deepcopy to save memory
                self.embedding_prediction_model = self.embedding_prediction_model.cpu()
                torch.cuda.empty_cache()  # Clear GPU cache
                self.embedding_prediction_model_ref = deepcopy(self.embedding_prediction_model)
                # Restore model to original device
                self.embedding_prediction_model = self.embedding_prediction_model.to(current_device)
            
            self.embedding_prediction_model_ref.eval()
            for param in self.embedding_prediction_model_ref.parameters():
                param.requires_grad = False
            # Keep reference model on CPU to save GPU memory
            # It will be moved to device only when needed in training_step
            self.embedding_prediction_model_ref = self.embedding_prediction_model_ref.cpu()
            
            # Load reference pre_trained_llm from path if specified, otherwise deepcopy current
            if self.reference_model_path is not None:
                log.info(f"Loading reference pre_trained_llm from: {self.reference_model_path}")
                checkpoint = torch.load(self.reference_model_path, map_location="cpu")
                # Temporarily move model to CPU for deepcopy to save memory
                self.pre_trained_llm = self.pre_trained_llm.cpu()
                torch.cuda.empty_cache()  # Clear GPU cache
                self.pre_trained_llm_ref = deepcopy(self.pre_trained_llm)
                # Restore model to original device
                self.pre_trained_llm = self.pre_trained_llm.to(current_device)
                
                state_dict = None
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif isinstance(checkpoint, dict) and any(k.startswith("model.") or "pre_trained_llm" in k for k in checkpoint.keys()):
                    state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Remove "model." prefix if present (PyTorch Lightning format)
                if state_dict and any(k.startswith("model.") for k in state_dict.keys()):
                    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
                
                self.pre_trained_llm_ref.load_state_dict(state_dict, strict=False)
                log.info("Reference pre_trained_llm loaded from checkpoint")
            else:
                log.info("Saving current pre_trained_llm as reference")
                # Temporarily move model to CPU for deepcopy to save memory
                self.pre_trained_llm = self.pre_trained_llm.cpu()
                torch.cuda.empty_cache()  # Clear GPU cache
                self.pre_trained_llm_ref = deepcopy(self.pre_trained_llm)
                # Restore model to original device
                self.pre_trained_llm = self.pre_trained_llm.to(current_device)
            
            self.pre_trained_llm_ref.eval()
            for param in self.pre_trained_llm_ref.parameters():
                param.requires_grad = False
            # Keep reference model on CPU to save GPU memory
            # It will be moved to device only when needed in training_step
            self.pre_trained_llm_ref = self.pre_trained_llm_ref.cpu()
            
            # Final cleanup
            torch.cuda.empty_cache()
        
        self.stage = stage
        if self.stage == "training":
            self._disable_grad(self.text_encoder)
            self._disable_grad(self.pre_trained_llm)
            self._enable_grad(self.embedding_prediction_model)
            self.pre_trained_llm.eval()
            self.text_encoder.eval()
            self.embedding_prediction_model.train()
        elif self.stage == "unlearning":
            self._disable_grad(self.text_encoder)
            self._enable_grad(self.embedding_prediction_model)
            self._enable_grad(self.pre_trained_llm)
            self.pre_trained_llm.train()
            self.embedding_prediction_model.eval()
            self.text_encoder.eval()

    def train(self, mode=True):
        if mode:
            super().train()
            if self.stage == "training":
                self.text_encoder.eval()
                self.pre_trained_llm.eval()
            elif self.stage == "unlearning":
                self.text_encoder.eval()
                self.embedding_prediction_model.eval()
        else:
            super().train(False)
        return self

    def update_unlearning_threshold(self, threshold: float):
        log.info(f"Updating unlearning threshold to {threshold}")
        self.unlearning_similarity_threshold = threshold

    def training_step(self, batch, batch_idx):
        # import pdb
        # pdb.set_trace()
        
        input_ids = batch["primary_input_ids"]  # shape (batch_size, max_length)
        context_windows = batch[
            "secondary_context_windows"
        ]  # shape (batch_size, max_length, context_window_length)
        has_full_window = batch["has_full_window"]  # shape (batch_size, max_length)
        attention_mask = batch["attention_mask"]  # shape (batch_size, max_length)

        batch_size, seq_len, window_len = context_windows.shape

        # Debug prints removed for cleaner output

        opt_list = self.optimizers()

        if self.stage == "training":
            # Forward pass through the reference encoder to get the target embeddings
            with torch.no_grad():
                reference_outputs = self.text_encoder(
                    input_ids=context_windows.view(-1, context_windows.size(-1)),
                    output_hidden_states=True,
                )
                attention_mask_enc = torch.ones_like(
                    context_windows.view(-1, context_windows.size(-1))
                )
                target_embeddings = mean_pooling_reference_encoder(
                    reference_outputs, attention_mask_enc
                ).view(batch_size, seq_len, -1)

            # Forward pass through the pretrained model
            with torch.no_grad():
                pretrained_outputs = self.pre_trained_llm(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    attention_mask=attention_mask,
                )
                hidden_states = pretrained_outputs.hidden_states[
                    self.hparams.pretrained_model_hook_layer
                ]

            # Forward pass through embedding prediction model
            outputs = self.embedding_prediction_model(hidden_states)
            loss = -torch.nn.functional.cosine_similarity(
                outputs, target_embeddings, dim=-1
            ) # shape (batch_size, max_length)
            loss = loss * has_full_window
            loss = loss.sum() / (has_full_window.sum() + 1e-8)
            assert not torch.isnan(loss), "Loss is NaN"

            opt_list[0].zero_grad()
            self.manual_backward(loss)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.embedding_prediction_model.parameters(),
                max_norm=self.hparams.clip_grad_norm,
            )
            self.log("train/grad_norm_embedding_clip", grad_norm, batch_size=batch_size)
            opt_list[0].step()

            if batch_idx == 0:
                assert not torch.all(outputs == 0), "Outputs are all zero"
                assert not torch.all(target_embeddings == 0), (
                    "Target embeddings are all zero"
                )

        elif self.stage == "unlearning":
            # Forward pass through the current pretrained model
            pretrained_outputs = self.pre_trained_llm(
                input_ids=input_ids, output_hidden_states=True, attention_mask=attention_mask
            )
            hidden_states = pretrained_outputs.hidden_states[
                self.hparams.pretrained_model_hook_layer
            ]

            # Forward pass through current embedding prediction model
            # ê = E_θ_E^*(x; θ_M) - current model generated embedding
            e_hat = self.embedding_prediction_model(
                hidden_states
            )  # shape (batch_size, seq_len, emb_dim)

            # Forward pass through reference models to get e_ref
            assert self.pre_trained_llm_ref is not None, (
                "Reference pre_trained_llm must be set before unlearning stage"
            )
            assert self.embedding_prediction_model_ref is not None, (
                "Reference embedding_prediction_model must be set before unlearning stage"
            )
            
            with torch.no_grad():
                # Reference models are kept on CPU to save GPU memory
                # Move inputs to CPU for reference model forward pass
                ref_device = next(self.pre_trained_llm_ref.parameters()).device
                input_ids_cpu = input_ids.to(ref_device)
                attention_mask_cpu = attention_mask.to(ref_device)
                
                # Get reference hidden states
                pretrained_outputs_ref = self.pre_trained_llm_ref(
                    input_ids=input_ids_cpu, output_hidden_states=True, attention_mask=attention_mask_cpu
                )
                hidden_states_ref = pretrained_outputs_ref.hidden_states[
                    self.hparams.pretrained_model_hook_layer
                ]
                
                # e_ref = E_θ_E^*(x; θ_M^ref) - reference embedding
                e_ref = self.embedding_prediction_model_ref(
                    hidden_states_ref
                )  # shape (batch_size, seq_len, emb_dim)
                
                # Move e_ref back to GPU for subsequent computations
                e_ref = e_ref.to(e_hat.device)

            # e_unlearn - unlearning target embedding
            # Ensure e_unlearn has the right shape: (1, 1, emb_dim) to broadcast with (batch_size, seq_len, emb_dim)
            e_unlearn = self.unlearning_target_embedding.detach().to(e_hat.device)
            # unlearning_target_embedding should be (emb_dim,) from mean_pooling_reference_encoder
            # We need (1, 1, emb_dim) for broadcasting
            if e_unlearn.dim() == 1:
                e_unlearn = e_unlearn.unsqueeze(0).unsqueeze(0)  # (emb_dim,) -> (1, 1, emb_dim)
            elif e_unlearn.dim() == 2:
                # If it's (1, emb_dim) or (seq_len, emb_dim), take mean or first
                if e_unlearn.shape[0] == 1:
                    e_unlearn = e_unlearn.unsqueeze(0)  # (1, emb_dim) -> (1, 1, emb_dim)
                else:
                    e_unlearn = e_unlearn.mean(dim=0, keepdim=True).unsqueeze(0)  # (seq_len, emb_dim) -> (1, 1, emb_dim)
            # Ensure it's exactly (1, 1, emb_dim)
            if e_unlearn.dim() != 3 or e_unlearn.shape[0] != 1 or e_unlearn.shape[1] != 1:
                # Fallback: reshape to (1, 1, emb_dim)
                e_unlearn = e_unlearn.view(1, 1, -1)

            # Compute similarities
            # sim(ê, e_unlearn) - similarity between current embedding and unlearning target
            sim_e_hat_unlearn = torch.nn.functional.cosine_similarity(
                e_hat, e_unlearn, dim=-1
            )  # shape (batch_size, seq_len)
            
            # sim(e_ref, e_unlearn) - similarity between reference embedding and unlearning target
            sim_e_ref_unlearn = torch.nn.functional.cosine_similarity(
                e_ref, e_unlearn, dim=-1
            )  # shape (batch_size, seq_len)

            # Δ(e) = sim(ê, e_unlearn) - sim(e_ref, e_unlearn)
            delta_e = sim_e_hat_unlearn - sim_e_ref_unlearn  # shape (batch_size, seq_len)
            
            # Store for logging
            self.delta_e = delta_e
            self.sim_e_hat_unlearn = sim_e_hat_unlearn
            self.sim_e_ref_unlearn = sim_e_ref_unlearn

            # L1: L_eNPO = - (1/β) E_x [log σ(-β Δ(e))]
            # Apply mask if needed (only compute loss where has_full_window is True)
            if "has_full_window" in batch:
                has_full_window = batch["has_full_window"]  # shape (batch_size, seq_len)
                # Only compute loss for valid positions
                valid_delta = delta_e * has_full_window
                # Compute eNPO loss
                l1_loss = - (1.0 / self.beta) * torch.log(
                    torch.sigmoid(-self.beta * valid_delta) + 1e-8
                )
                # Average over valid positions
                l1_loss = l1_loss.sum() / (has_full_window.sum() + 1e-8)
            else:
                # Compute eNPO loss for all positions
                l1_loss = - (1.0 / self.beta) * torch.log(
                    torch.sigmoid(-self.beta * delta_e) + 1e-8
                ).mean()
            
            assert not torch.isnan(l1_loss), "L1 loss is NaN"
            
            # L2: KL divergence loss to maintain non-forgetting capability
            # Compute logits from current model and reference model
            # Get logits from current model (already computed in pretrained_outputs)
            current_logits = pretrained_outputs.logits  # shape (batch_size, seq_len, vocab_size)
            
            # Get logits from reference model
            with torch.no_grad():
                ref_device = next(self.pre_trained_llm_ref.parameters()).device
                input_ids_cpu = input_ids.to(ref_device)
                attention_mask_cpu = attention_mask.to(ref_device)
                
                # Get reference logits
                pretrained_outputs_ref_logits = self.pre_trained_llm_ref(
                    input_ids=input_ids_cpu,
                    attention_mask=attention_mask_cpu,
                )
                ref_logits = pretrained_outputs_ref_logits.logits  # shape (batch_size, seq_len, vocab_size)
                ref_logits = ref_logits.to(current_logits.device)
            
            # Compute KL divergence: KL(P_current || P_ref)
            # Apply temperature to stabilize training (optional, using 1.0 by default)
            temperature = 1.0
            current_log_probs = torch.nn.functional.log_softmax(current_logits / temperature, dim=-1)
            ref_log_probs = torch.nn.functional.log_softmax(ref_logits / temperature, dim=-1)
            
            # KL divergence: sum over vocabulary dimension
            kl_div = torch.nn.functional.kl_div(
                current_log_probs.view(-1, current_log_probs.size(-1)),
                ref_log_probs.view(-1, ref_log_probs.size(-1)),
                reduction='none',
                log_target=True
            ).sum(dim=-1)  # shape (batch_size * seq_len,)
            
            # Reshape back to (batch_size, seq_len)
            kl_div = kl_div.view(batch_size, seq_len)
            
            # Apply attention mask to ignore padding tokens
            if attention_mask is not None and isinstance(attention_mask, torch.Tensor) and attention_mask.dim() >= 2:
                # Ensure attention_mask is a 2D tensor
                if attention_mask.dim() == 2:
                    # Check if sequence lengths match
                    if kl_div.shape[1] == attention_mask.shape[1]:
                        # Use attention mask directly
                        kl_mask = attention_mask.float()
                    else:
                        # Handle sequence length mismatch
                        min_len = min(kl_div.shape[1], attention_mask.shape[1])
                        kl_mask = attention_mask[:, :min_len].float()
                        kl_div = kl_div[:, :min_len]
                    
                    # Average over valid tokens only
                    l2_loss = (kl_div * kl_mask).sum() / (kl_mask.sum() + 1e-8)
                else:
                    # If attention_mask has unexpected dimensions, use mean
                    log.warning(f"Unexpected attention_mask shape: {attention_mask.shape}, using mean for L2 loss")
                    l2_loss = kl_div.mean()
            else:
                # Average over all positions if no valid attention mask
                l2_loss = kl_div.mean()
            
            assert not torch.isnan(l2_loss), "L2 loss is NaN"
            
            # Total loss: weighted sum of L1 and L2
            loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
            
            assert not torch.isnan(loss), "Total loss is NaN"
            
            # Store L1 and L2 losses for logging
            self.l1_loss = l1_loss
            self.l2_loss = l2_loss

            opt_list[1].zero_grad()
            self.manual_backward(loss)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.pre_trained_llm.parameters(), max_norm=self.hparams.clip_grad_norm
            )
            self.log("train/grad_norm_pre_clip", grad_norm, batch_size=batch_size)
            opt_list[1].step()
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

        self.log("train/loss", loss, batch_size=batch_size)
        if self.stage == "training":
            self.log("train/training_loss", loss, batch_size=batch_size)
        else:
            self.log("train/unlearning_loss", loss, batch_size=batch_size)
            self.log("train/enpo_beta", self.beta)
            # Log L1 and L2 losses separately
            if hasattr(self, 'l1_loss'):
                self.log("train/l1_loss", self.l1_loss, batch_size=batch_size)
            if hasattr(self, 'l2_loss'):
                self.log("train/l2_loss", self.l2_loss, batch_size=batch_size)
            self.log("train/l1_weight", self.l1_weight)
            self.log("train/l2_weight", self.l2_weight)
            # Log additional metrics for debugging
            if hasattr(self, 'delta_e'):
                self.log("train/delta_e_mean", self.delta_e.mean(), batch_size=batch_size)
                self.log("train/sim_e_hat_unlearn_mean", self.sim_e_hat_unlearn.mean(), batch_size=batch_size)
                self.log("train/sim_e_ref_unlearn_mean", self.sim_e_ref_unlearn.mean(), batch_size=batch_size)
        return {"loss": loss}

    def configure_optimizers(self):
        return [
            torch.optim.Adam(
                self.embedding_prediction_model.parameters(),
                lr=self.hparams.training_lr,
                weight_decay=self.hparams.training_weight_decay,
            ),
            torch.optim.SGD(
                self.pre_trained_llm.parameters(),
                lr=self.hparams.unlearning_lr,
                weight_decay=self.hparams.unlearning_weight_decay,
            ),
        ]
