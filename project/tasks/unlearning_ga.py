from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment
from project.eval import eval_llm
from hydra.utils import instantiate
import torch
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from project.utils import get_logger, log_hyperparameters
from lightning.pytorch.loggers import Logger
from project.utils.callbacks import get_default_callbacks
from pathlib import Path

log = get_logger()

# Try to import PEFT for LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    log.warning("PEFT library not available. LoRA support will be disabled. Install with: pip install peft")


class UnlearningGA:
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
        log.info("Task: unlearning_ga")
        task_config = self.global_config.task

        log.info("Loading data")
        unlearning_datamodule = instantiate(
            self.task_config.unlearning_data,
            primary_tokenizer=self.pre_trained_llm_tokenizer,
            target_ids=[self.target_id],
        )
        unlearning_datamodule.prepare_data()
        unlearning_datamodule.setup("train")

        log.info("Instantiating UnlearningGATrainingModule")
        task = UnlearningGATrainingModule(
            pre_trained_llm=self.pre_trained_llm,
            pre_trained_llm_tokenizer=self.pre_trained_llm_tokenizer,
            **task_config.training_module,
        )

        log_hyperparameters(self.logger, self.global_config, [("pre_trained_llm", self.pre_trained_llm)])

        log.info("Instantiating trainer")
        trainer = Trainer(
            **self.global_config.trainer,
            callbacks=get_default_callbacks(),
            logger=self.logger,
            plugins=[SLURMEnvironment(auto_requeue=False)],
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
        # Get save configuration
        save_unlearned_model = getattr(self.task_config, "save_unlearned_model", False)
        save_dir = None
        if save_unlearned_model:
            save_dir = Path("/root/autodl-tmp/saved-models") / "ga" / self.target_id
            save_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"模型将保存到: {save_dir}")
        
        last_unlearning_idx = len(task_config.stages) - 1
        for idx, stage in enumerate(task_config.stages):
            assert stage["type"] == "unlearning"
            log.info(
                f"Starting stage {idx + 1} ({stage['type']}) of {len(task_config.stages)}"
            )
            new_max_steps = (
                stage["steps"] if idx == 0 else stage["steps"] + trainer.max_steps
            )
            log.info(f"Setting max steps to {new_max_steps}")
            trainer.fit_loop.epoch_loop.max_steps = new_max_steps
            trainer.fit(task, datamodule=unlearning_datamodule)
            log.info(f"Stage {idx + 1} ({stage['type']}) completed!")
            log.info("Starting testing!")
            results = eval_llm(
                self.pre_trained_llm,
                self.pre_trained_llm_tokenizer,
                self.target_id,
                trainer.strategy.root_device,
                0,
            )
            trainer.logger.log_metrics(results)
            
            # Save model after last stage if configured
            if idx == last_unlearning_idx and save_unlearned_model:
                unlearned_model_path = save_dir / "unlearned_model.pt"
                log.info(f"保存被遗忘的目标模型到: {unlearned_model_path}")
                original_device = next(self.pre_trained_llm.parameters()).device
                model_to_save = task.pre_trained_llm
                model_to_save = model_to_save.cpu()
                
                if task.use_lora and task.lora_loaded and PEFT_AVAILABLE:
                    # Save LoRA weights
                    lora_path = save_dir / "unlearned_model_lora"
                    lora_path.mkdir(parents=True, exist_ok=True)
                    model_to_save.save_pretrained(str(lora_path))
                    log.info(f"LoRA权重已保存到: {lora_path}")
                    torch.save({
                        "lora_path": str(lora_path),
                        "use_lora": True,
                        "state_dict": model_to_save.state_dict(),
                    }, unlearned_model_path)
                else:
                    # Save full model
                    torch.save(model_to_save.state_dict(), unlearned_model_path)
                
                task.pre_trained_llm = model_to_save.to(original_device)
                self.pre_trained_llm = task.pre_trained_llm
                log.info(f"被遗忘的目标模型已保存")
        log.info("Unlearning complete!")


class UnlearningGATrainingModule(pl.LightningModule):
    def __init__(
        self,
        pre_trained_llm: AutoModelForCausalLM,
        pre_trained_llm_tokenizer: AutoTokenizer,
        lr: float,
        weight_decay: float,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("pre_trained_llm", "pre_trained_llm_tokenizer"))
        self.pre_trained_llm = pre_trained_llm
        self.pre_trained_llm_tokenizer = pre_trained_llm_tokenizer
        self.automatic_optimization = True
        
        # LoRA configuration
        self.use_lora = kwargs.get("use_lora", False)
        self.lora_config = kwargs.get("lora_config", None)
        self.lora_loaded = False
        
        # Apply LoRA if enabled
        if self.use_lora:
            if not PEFT_AVAILABLE:
                log.warning("LoRA requested but PEFT library not available. Falling back to full fine-tuning.")
                self.use_lora = False
            else:
                log.info("Applying LoRA to pre_trained_llm")
                if self.lora_config is None:
                    # Default LoRA configuration
                    log.info("Using default LoRA configuration")
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=16,
                        lora_alpha=32,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        bias="none",
                    )
                else:
                    # Use provided LoRA configuration
                    # Convert OmegaConf DictConfig to regular dict if needed
                    from omegaconf import DictConfig as OmegaDictConfig
                    if isinstance(self.lora_config, OmegaDictConfig):
                        lora_config_dict = OmegaConf.to_container(self.lora_config, resolve=True)
                        lora_config = LoraConfig(**lora_config_dict)
                    elif isinstance(self.lora_config, dict):
                        lora_config = LoraConfig(**self.lora_config)
                    else:
                        lora_config = self.lora_config
                
                # Apply LoRA to the model
                self.pre_trained_llm = get_peft_model(self.pre_trained_llm, lora_config)
                self.lora_loaded = True
                log.info(f"LoRA applied. Trainable parameters: {self.pre_trained_llm.get_nb_trainable_parameters()}")
        
        if not self.use_lora:
            # Full fine-tuning: enable gradients for all parameters
            for param in self.pre_trained_llm.parameters():
                param.requires_grad = True
        else:
            # LoRA: only LoRA parameters have gradients enabled (handled by PEFT)
            # Disable gradients for base model parameters
            for name, param in self.pre_trained_llm.named_parameters():
                if "lora" not in name.lower():
                    param.requires_grad = False

    def training_step(self, batch, batch_idx):
        input_ids = batch["primary_input_ids"]  # shape (batch_size, max_length)
        attention_mask = batch["attention_mask"]  # shape (batch_size, max_length)
        labels = batch["primary_labels"]  # shape (batch_size, max_length)

        batch_size, seq_len = input_ids.shape

        outputs = self.pre_trained_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = -outputs.loss  # gradient ascent

        self.log("train/loss", loss, batch_size=batch_size)

        return {"loss": loss}

    def configure_optimizers(self):
        if self.use_lora and self.lora_loaded:
            # Only optimize trainable (LoRA) parameters
            trainable_params = [p for p in self.pre_trained_llm.parameters() if p.requires_grad]
            log.info(f"Configuring optimizer for {len(trainable_params)} LoRA parameter groups")
            return [
                torch.optim.AdamW(
                    trainable_params,
                    lr=self.hparams.lr,
                    weight_decay=self.hparams.weight_decay,
                ),
            ]
        else:
            # Full fine-tuning: optimize all parameters
            return [
                torch.optim.AdamW(
                    self.pre_trained_llm.parameters(),
                    lr=self.hparams.lr,
                    weight_decay=self.hparams.weight_decay,
                ),
            ]
