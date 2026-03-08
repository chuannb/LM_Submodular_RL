"""
DPO / ORPO Trainer

Fine-tune Qwen3-Reranker hoặc bất kỳ causal LM nào trên preference data
bằng DPO (Direct Preference Optimisation) hoặc ORPO (Odds Ratio PO).

Sử dụng TRL (Transformer Reinforcement Learning) library của HuggingFace.

Input data (JSONL):  {"prompt": ..., "chosen": ..., "rejected": ...}

DPO loss:
  L_DPO = -log σ(β * (log π(chosen|x)/π_ref(chosen|x)
                     - log π(rejected|x)/π_ref(rejected|x)))

ORPO loss (không cần reference model – rẻ hơn):
  L_ORPO = L_SFT + λ * L_OR
  L_OR   = -log σ(log(odds(chosen)) - log(odds(rejected)))
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    """
    HuggingFace-compatible Dataset for DPO/ORPO.
    Each sample: {"prompt": str, "chosen": str, "rejected": str}
    """

    def __init__(self, jsonl_path: str):
        self.samples: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# DPO Trainer wrapper
# ---------------------------------------------------------------------------

class DPOFinetuner:
    """
    Fine-tune a causal LM (e.g. Qwen3-Reranker backbone) using DPO.

    Uses TRL's DPOTrainer under the hood.

    Parameters
    ----------
    model_id     : base model to fine-tune
    device       : "cpu" | "cuda"
    beta         : KL regularisation coefficient (DPO β)
    use_lora     : use LoRA to reduce memory
    lora_r       : LoRA rank
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "cpu",
        beta: float = 0.1,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        self.model_id = model_id
        self.device = device
        self.beta = beta
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

    def _load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if self.device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        if self.use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                )
                model = get_peft_model(model, lora_cfg)
                model.print_trainable_parameters()
            except ImportError:
                print("peft not installed, using full fine-tuning.")

        return model, tokenizer

    # ------------------------------------------------------------------
    def train_dpo(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        output_dir: str = "checkpoints/dpo",
        num_epochs: int = 3,
        batch_size: int = 2,
        gradient_accumulation: int = 8,
        lr: float = 5e-7,
        max_length: int = 1024,
        max_prompt_length: int = 512,
    ) -> None:
        """Fine-tune with DPO using TRL's DPOTrainer."""
        try:
            from trl import DPOTrainer, DPOConfig
        except ImportError as e:
            raise ImportError("Install trl: pip install trl>=0.9.0") from e

        model, tokenizer = self._load_model_and_tokenizer()
        train_ds = PreferenceDataset(train_path)
        val_ds = PreferenceDataset(val_path) if val_path else None

        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=lr,
            beta=self.beta,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if val_ds else "no",
            fp16=(self.device != "cpu"),
            remove_unused_columns=False,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,   # None = implicit reference via β-regularisation (efficient)
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
        )

        print(f"Starting DPO training (β={self.beta})...")
        trainer.train()
        trainer.save_model(os.path.join(output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(output_dir, "final"))
        print(f"DPO training complete -> {output_dir}/final")

    # ------------------------------------------------------------------
    def train_orpo(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        output_dir: str = "checkpoints/orpo",
        num_epochs: int = 3,
        batch_size: int = 2,
        gradient_accumulation: int = 8,
        lr: float = 8e-6,
        lambda_orpo: float = 0.1,
        max_length: int = 1024,
    ) -> None:
        """
        Fine-tune with ORPO (no reference model needed → cheaper).
        Uses TRL's ORPOTrainer.
        """
        try:
            from trl import ORPOTrainer, ORPOConfig
        except ImportError as e:
            raise ImportError("Install trl>=0.9.0: pip install trl>=0.9.0") from e

        model, tokenizer = self._load_model_and_tokenizer()
        train_ds = PreferenceDataset(train_path)
        val_ds = PreferenceDataset(val_path) if val_path else None

        training_args = ORPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=lr,
            lambda_=lambda_orpo,
            max_length=max_length,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if val_ds else "no",
            fp16=(self.device != "cpu"),
            remove_unused_columns=False,
        )

        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
        )

        print(f"Starting ORPO training (λ={lambda_orpo})...")
        trainer.train()
        trainer.save_model(os.path.join(output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(output_dir, "final"))
        print(f"ORPO training complete -> {output_dir}/final")
