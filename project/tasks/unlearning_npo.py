from copy import deepcopy
from pathlib import Path
from omegaconf import DictConfig
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

log = get_logger()


class UnlearningNPO:
    def __init__(
        self,
        global_config: DictConfig,
        target_id: str,
        target_name: str,
        pre_trained_llm: AutoModelForCausalLM,
        pre_trained_llm_tokenizer: AutoTokenizer,
        logger: Logger,
        beta: float,
        **kwargs,
    ):
        self.global_config = global_config
        self.task_config = global_config.task
        self.target_id = target_id
        self.target_name = target_name
        self.pre_trained_llm = pre_trained_llm
        self.pre_trained_llm_tokenizer = pre_trained_llm_tokenizer
        self.logger = logger
        self.beta = beta

    def unlearn(self):
        log.info("Task: unlearning_npo")
        task_config = self.global_config.task

        log.info("Creating reference model (model_ref)")
        # 将模型移到CPU进行deepcopy，避免占用GPU显存
        original_device = next(self.pre_trained_llm.parameters()).device
        log.info(f"Moving model to CPU for deepcopy (original device: {original_device})")
        self.pre_trained_llm = self.pre_trained_llm.cpu()
        torch.cuda.empty_cache()  # 清空GPU缓存
        
        model_ref = deepcopy(self.pre_trained_llm)
        model_ref.eval()
        for param in model_ref.parameters():
            param.requires_grad = False
        
        # 将model_ref保持在CPU上，model_theta移回原设备
        model_ref = model_ref.cpu()
        self.pre_trained_llm = self.pre_trained_llm.to(original_device)
        log.info(f"model_ref created and kept on CPU, model_theta moved back to {original_device}")

        log.info("Loading data")
        unlearning_datamodule = instantiate(
            self.task_config.unlearning_data,
            primary_tokenizer=self.pre_trained_llm_tokenizer,
            target_ids=[self.target_id],
        )
        unlearning_datamodule.prepare_data()
        unlearning_datamodule.setup("train")

        log.info("Instantiating UnlearningNPOTrainingModule")
        task_module = UnlearningNPOTrainingModule(
            model_theta=self.pre_trained_llm,
            model_ref=model_ref,
            pre_trained_llm_tokenizer=self.pre_trained_llm_tokenizer,
            beta=self.beta,
            **task_config.training_module,
        )

        log_hyperparameters(self.logger, self.global_config, [("model_theta", self.pre_trained_llm)])

        log.info("Instantiating trainer")
        trainer = Trainer(
            **self.global_config.trainer,
            callbacks=get_default_callbacks(enable_checkpointing=False),
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
        for idx, stage in enumerate(task_config.stages):
            assert stage["type"] == "unlearning"
            log.info(
                f"Starting stage {idx + 1} ({stage['type']}) of {len(task_config.stages)}"
            )
            current_max_steps = trainer.max_steps if trainer.max_steps is not None and trainer.max_steps > 0 else 0
            if idx == 0 :
                 new_max_steps = stage["steps"]
            else:
                 new_max_steps = current_max_steps + stage["steps"]

            log.info(f"Setting max steps to {new_max_steps}")
            trainer.fit_loop.epoch_loop.max_steps = new_max_steps
            trainer.fit(task_module, datamodule=unlearning_datamodule)
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
        log.info("Unlearning complete!")
        
        # 保存被遗忘的模型
        save_dir = Path("/root/autodl-tmp/saved-models") / "npo" / self.target_id
        save_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"模型将保存到: {save_dir}")
        
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


class UnlearningNPOTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model_theta: AutoModelForCausalLM,
        model_ref: AutoModelForCausalLM,
        pre_trained_llm_tokenizer: AutoTokenizer,
        lr: float,
        weight_decay: float,
        beta: float,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("model_theta", "model_ref", "pre_trained_llm_tokenizer"))
        self.model_theta = model_theta
        self.model_ref = model_ref
        self.pre_trained_llm_tokenizer = pre_trained_llm_tokenizer
        self.automatic_optimization = True

        for param in self.model_theta.parameters():
            param.requires_grad = True
        
        # 启用梯度检查点以节省显存
        if hasattr(self.model_theta, 'gradient_checkpointing_enable'):
            self.model_theta.gradient_checkpointing_enable()
            log.info("Enabled gradient checkpointing to save GPU memory")
        
        self.model_ref.eval()
        for param in self.model_ref.parameters():
            param.requires_grad = False
        
        # 确保model_ref在CPU上
        if next(self.model_ref.parameters()).device.type != 'cpu':
            log.info("Moving model_ref to CPU to save GPU memory")
            self.model_ref = self.model_ref.cpu()
    
    def on_fit_start(self):
        """在训练开始前确保model_ref在CPU上"""
        if next(self.model_ref.parameters()).device.type != 'cpu':
            log.info("Moving model_ref to CPU before training starts")
            self.model_ref = self.model_ref.cpu()
            torch.cuda.empty_cache()
    
    def on_train_batch_start(self, batch, batch_idx):
        """在每个训练batch开始前确保model_ref在CPU上"""
        # Lightning可能会自动移动模型，所以每次都要检查
        ref_device = next(self.model_ref.parameters()).device
        if ref_device.type != 'cpu':
            log.warning(f"model_ref was moved to {ref_device}, moving back to CPU")
            self.model_ref = self.model_ref.cpu()
            torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        input_ids = batch["primary_input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["primary_labels"]

        batch_size = input_ids.shape[0]

        # 前向传播model_theta
        outputs_theta = self.model_theta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        log_probs_theta = -outputs_theta.loss

        # 在CPU上运行model_ref的前向传播，避免占用GPU显存
        # 将输入移到CPU，计算后再移回GPU
        ref_device = next(self.model_ref.parameters()).device
        if ref_device.type != 'cpu':
            # 如果model_ref不在CPU上，先移回CPU
            self.model_ref = self.model_ref.cpu()
            torch.cuda.empty_cache()
        
        # 将输入移到CPU进行计算
        input_ids_cpu = input_ids.cpu()
        attention_mask_cpu = attention_mask.cpu()
        labels_cpu = labels.cpu()
        
        with torch.no_grad():
            outputs_ref = self.model_ref(
                input_ids=input_ids_cpu,
                attention_mask=attention_mask_cpu,
                labels=labels_cpu,
            )
            log_probs_ref = -outputs_ref.loss
        
        # 将结果移回GPU（log_probs_ref是一个标量，占用很少显存）
        log_probs_ref = log_probs_ref.to(input_ids.device)
        
        # 定期清空缓存
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        npo_loss_term = -self.hparams.beta * (log_probs_theta - log_probs_ref)
        loss = -(2 / self.hparams.beta) * torch.nn.functional.logsigmoid(npo_loss_term)
        
        self.log("train/loss", loss, batch_size=batch_size, prog_bar=True)
        self.log("train/log_probs_theta", log_probs_theta, batch_size=batch_size)
        self.log("train/log_probs_ref", log_probs_ref, batch_size=batch_size)

        return {"loss": loss}

    def configure_optimizers(self):
        return [
            torch.optim.AdamW(
                self.model_theta.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            ),
        ]
