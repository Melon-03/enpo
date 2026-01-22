from copy import deepcopy
from typing import Any

import lightning.pytorch as pl
import torch
from lightning import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint
import wandb
from .logging import get_logger

log = get_logger()

def get_default_callbacks(enable_checkpointing: bool = True):
    """获取默认的callbacks列表。
    
    Args:
        enable_checkpointing: 如果为False，则不包含ModelCheckpoint callback
    """
    callbacks = [
        WandbSummaries(monitor="eval/forget/fb", mode="min"),
        TQDMProgressBar(refresh_rate=1),
    ]
    
    if enable_checkpointing:
        # 配置checkpoint callback以节省磁盘空间
        # 只保存模型权重，不保存优化器状态等
        # 只保留最新的1个checkpoint
        # 保存到指定目录以使用更大容量的存储空间
        callbacks.insert(0, ModelCheckpoint(
            dirpath="/root/autodl-tmp/checkpoints-enpo",  # 指定checkpoint保存目录
            save_weights_only=True,  # 只保存模型权重，不保存优化器状态等（节省约50%空间）
            save_top_k=1,  # 只保留最新的1个checkpoint
            every_n_train_steps=1000,  # 每5000步保存一次
            save_on_train_epoch_end=True,  # 训练结束时也保存（确保训练完成时有checkpoint）
        ))
        callbacks.insert(1, SelectiveCheckpoint())  # 根据stage只保存可训练的模型，节省大量空间
    
    return callbacks


class AlwaysSaveCheckpoints(Callback):
    """Log model checkpoints even if training failed.

    As of 04/09/2024, WandbLogger only saves checkpoints on successful runs.
    """

    def on_exception(self, trainer, pl_module, exception):
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger._scan_and_log_checkpoints(logger._checkpoint_callback)


class SelectiveCheckpoint(Callback):
    """只保存可训练模型的checkpoint，节省磁盘空间。
    
    训练阶段：只保存 embedding_prediction_model (~几MB，而不是3.8GB的LLM)
    Unlearning阶段：保存 pre_trained_llm 和 embedding_prediction_model
    """
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict[str, Any]):
        """过滤checkpoint，只保留可训练的模型"""
        if not hasattr(pl_module, 'stage'):
            # 如果没有stage属性，保存所有模型（向后兼容）
            return
        
        stage = pl_module.stage
        state_dict = checkpoint.get("state_dict", {})
        filtered_state_dict = {}
        
        if stage == "training":
            # 训练阶段：只保存 embedding_prediction_model
            # LLM和text_encoder都是冻结的，不需要保存
            for key, value in state_dict.items():
                if "embedding_prediction_model" in key:
                    filtered_state_dict[key] = value
            log.info(f"训练阶段：只保存 embedding_prediction_model")
        elif stage == "unlearning":
            # Unlearning阶段：保存 pre_trained_llm 和 embedding_prediction_model
            # text_encoder是冻结的，不需要保存
            for key, value in state_dict.items():
                if "pre_trained_llm" in key or "embedding_prediction_model" in key:
                    filtered_state_dict[key] = value
            log.info("Unlearning阶段：保存 pre_trained_llm 和 embedding_prediction_model")
        
        checkpoint["state_dict"] = filtered_state_dict


class ConfigInCheckpoint(Callback):
    """Save the config in the checkpoint."""

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict[str, Any]):
        checkpoint["config"] = OmegaConf.to_container(self.config, resolve=True)


class WandbSummaries(pl.Callback):
    """Set the W&B summaries of each metric to the values from the best epoch."""

    def __init__(self, monitor: str, mode: str):
        super().__init__()

        self.monitor = monitor
        self.mode = mode

        self.best_metric = None
        self.best_metrics = None

        self.ready = True

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ready = False

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ready = True

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.ready:
            return

        metrics = trainer.logged_metrics
        if self.monitor in metrics:
            metric = metrics[self.monitor]
            if torch.is_tensor(metric):
                metric = metric.item()

            if self._better(metric):
                self.best_metric = metric
                self.best_metrics = deepcopy(metrics)

        self._update_summaries()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._update_summaries()

    def state_dict(self):
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best_metric": self.best_metric,
            "best_metrics": self.best_metrics,
        }

    def load_state_dict(self, state_dict):
        self.monitor = state_dict["monitor"]
        self.mode = state_dict["mode"]
        self.best_metric = state_dict["best_metric"]
        self.best_metrics = state_dict["best_metrics"]

    def _better(self, metric):
        if self.best_metric is None:
            return True
        elif self.mode == "min" and metric < self.best_metric:
            return True
        elif self.mode == "max" and metric > self.best_metric:
            return True
        else:
            return False

    def _update_summaries(self):
        # wandb is supposed not to update the summaries anymore once we set them manually,
        # but they are still getting updated, so we make sure to set them after logging
        if self.best_metrics is not None:
            wandb.summary.update(self.best_metrics)
