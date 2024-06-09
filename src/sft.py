from dataclasses import dataclass, field
from typing import Optional

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
from trl import (
    ModelConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map, TrlParser,
)


@dataclass
class ScriptArguments:
    train_dataset_path: str = field(metadata={"help": "Path to a CSV file to be used as training data"})
    val_dataset_path: str = field(metadata={"help": "Path to a CSV file to be used as validation data"})
    prompt_format: Optional[str] = field(default="Q: {q}\nA: {a}\nTrue: {s:.1f}", metadata={"help": "Q/A/score format"})
    max_seq_length: Optional[int] = field(default=256, metadata={"help": "Maximum number of tokens in an example"})
    wandb_project_name: Optional[str] = field(default="llm-calib", metadata={"help": "Q/A/score format"})


def main(script_args: ScriptArguments, training_args: TrainingArguments, model_config: ModelConfig):
    if "wandb" in training_args.report_to:
        wandb.init(project=script_args.wandb_project_name)

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset_text_field = "text"

    def formatting_prompts_func(examples):
        texts = []
        for q, a, s in zip(examples['question'], examples['pred_answer'], examples['score']):
            texts.append(script_args.prompt_format.format(q=q, a=a, s=s) + tokenizer.eos_token)

        return {dataset_text_field: texts}

    train_dataset = load_dataset('csv', data_files=script_args.train_dataset_path, split='train')
    eval_dataset = load_dataset('csv', data_files=script_args.val_dataset_path, split='train')

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        dataset_text_field=dataset_text_field,
        max_seq_length=script_args.max_seq_length,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=True,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments, ModelConfig))  # noqa
    main(*parser.parse_args_and_config())
