import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

import evaluate
import numpy as np
import pandas as pd
import timeout_decorator
import torch
import wandb
from datasets import load_dataset, Dataset
from evaluate import EvaluationModule
from scipy.stats import pearsonr
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, ModelConfig, get_peft_config, \
    get_quantization_config
from trl.commands.cli_utils import TrlParser

from src.postprocess_dataset import normalise_answer


@dataclass
class ScriptArguments:
    train_dataset_path: str = field(metadata={"help": "Path to a CSV file to be used as training data"})
    val_dataset_path: str = field(metadata={"help": "Path to a CSV file to be used as validation data"})
    save_root: str = field(metadata={"help": "Root path for saving the checkpoints"})
    save_freq: Optional[int] = field(default=10, metadata={"help": "Number of iterations between checkpoint saves"})
    val_freq: Optional[int] = field(default=20, metadata={"help": "Number of iterations between validation runs"})
    val_dataset_size: Optional[int] = field(default=512, metadata={"help": "Number of examples to use during validation"})
    score_lambda: Optional[float] = field(default=0.1, metadata={"help": "Oracle score coefficient in reward function"})
    oracle_model: Optional[str] = field(default="BLEURT-20", metadata={"help": "Type of BLEURT model to use"})
    prompt_format: Optional[str] = field(default="Q: {question}\nA:", metadata={"help": "Q/A prompt format"})
    max_pred_tokens: Optional[int] = field(default=64, metadata={"help": "Maximum number of tokens to predict"})
    malformed_resp_reward: Optional[float] = field(default=-2.0, metadata={"help": "Reward assigned to malformed responses"})
    max_epochs: Optional[int] = field(default=4, metadata={"help": "Maximum number of epochs to train for"})


def load_and_tokenize_dataset(tokenizer: PreTrainedTokenizerBase, dataset_path: str, prompt_format: str) -> Dataset:
    dataset = load_dataset('csv', data_files=dataset_path, split='train')

    def tokenize(example):
        prompt = prompt_format.format(question=example["question"])
        example["input_ids"] = tokenizer(prompt)['input_ids']
        example["query"] = tokenizer.decode(example["input_ids"])
        return example

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format("torch")
    return dataset


def infer_single(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_pred_tokens: int,
    device: str = "cuda"
) -> str:
    question_tokens = tokenizer([question,], return_tensors="pt").to(device)
    prompt_length = question_tokens['input_ids'].shape[1]

    outputs = model.generate(**question_tokens, do_sample=False, max_new_tokens=max_pred_tokens)
    return tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)


def parse_answer(resp: str, debug: bool = False) -> Tuple[Optional[str], Optional[float]]:
    resp_lines = resp.split("\n")

    if len(resp_lines) != 2:
        if debug:
            print("resp", resp)
        return resp, None

    pred = resp_lines[0]

    try:
        confidence_str = resp_lines[1].strip().replace("True: ", "")
        confidence_raw = float(confidence_str)
        confidence = np.clip(confidence_raw, 0, 1)
    except Exception:
        if debug:
            print("resp_lines[1]", resp_lines[1])
        return pred, None

    return pred, confidence


def eval_batch(
    responses: List[str],
    gt_answers: List[str],
    oracle: EvaluationModule,
    score_lambda: float,
    malformed_response_reward: float
) -> Tuple[List[torch.FloatTensor], List[tuple]]:
    assert len(responses) == len(gt_answers)

    rewards = []
    scatter_log_data = []
    for resp, gt_resp in zip(responses, gt_answers):
        pred_text, pred_conf = parse_answer(resp)

        if pred_conf is None:
            reward_float = malformed_response_reward

        else:
            pred_normalised = normalise_answer(pred_text)
            gt_normalised = normalise_answer(gt_resp)

            if pred_normalised == gt_normalised:
                score_raw = 1.0

            else:
                score_raw = oracle.compute(predictions=[pred_normalised], references=[gt_normalised])["scores"][0]

            score = np.clip(score_raw, 0, 1)
            reward_float = score_lambda * score - (score - pred_conf) ** 2
            scatter_log_data.append((pred_conf, score, gt_resp, pred_text))

        rewards.append(torch.FloatTensor([reward_float]))

    assert len(rewards) == len(responses)

    return rewards, scatter_log_data


def collator(data):
    output = {key: [d[key] for d in data] for key in data[0]}
    return output


def calculate_and_log_metrics(scatter_log_data: List[tuple], prefix: str = "", commit: bool = False):
    table = wandb.Table(data=scatter_log_data, columns=["llm_confidence", "oracle_score", "gt_resp", "pred"])

    llm_confidences = [x[0] for x in scatter_log_data]
    oracle_scores = [x[1] for x in scatter_log_data]

    corr = pearsonr(oracle_scores, llm_confidences)[0]
    mean_confidence = np.mean(llm_confidences)
    mean_score = np.mean(oracle_scores)

    print(f"{prefix}mean_confidence", mean_confidence)
    print(f"{prefix}mean_score", mean_score)
    print(f"{prefix}corr", corr)

    wandb.log({
        f"{prefix}calibration": wandb.plot.scatter(table, "llm_confidence", "oracle_score"),
        f"{prefix}llm_confidence": wandb.Histogram(llm_confidences),
        f"{prefix}oracle_score": wandb.Histogram(oracle_scores),
        f"{prefix}mean_confidence": mean_confidence,
        f"{prefix}mean_score": mean_score,
        f"{prefix}pearsonr": corr,
    }, commit=commit)


@timeout_decorator.timeout(10, timeout_exception=TimeoutError)
def save_with_timeout(trainer: PPOTrainer, save_directory: str):
    trainer.save_pretrained(save_directory)


def main(script_args: ScriptArguments, config: PPOConfig, model_config: ModelConfig):
    oracle = evaluate.load("bleurt", config_name=script_args.oracle_model)

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    quantization_config = get_quantization_config(model_config)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map={"": "cuda"},
        peft_config=get_peft_config(model_config),
        attn_implementation=model_config.attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_and_tokenize_dataset(tokenizer, script_args.train_dataset_path, script_args.prompt_format)

    ppo_trainer = PPOTrainer(
        config,
        model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )

    generation_kwargs = {
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": script_args.max_pred_tokens,

        "min_length": -1,  # don't ignore the EOS token
        "top_k": 0.0,  # no top-k sampling
        "top_p": 1.0,  # no nucleus sampling
    }

    save_dir = os.path.join(script_args.save_root, config.model_name)
    os.makedirs(save_dir, exist_ok=True)

    df_valid = pd.read_csv(script_args.val_dataset_path)
    validation_set = list(df_valid.iterrows())[:script_args.val_dataset_size]

    global_iter = 0
    for epoch in range(script_args.max_epochs):
        for batch_idx, batch in enumerate(ppo_trainer.dataloader):
            question_tensors = batch["input_ids"]

            response_tensors = ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                **generation_kwargs,
            )

            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            print("-" * 80)
            print("epoch", epoch)
            print("batch", batch_idx)

            rewards, scatter_log_data = eval_batch(
                batch["response"],
                batch["gt_answer"],
                oracle,
                script_args.score_lambda,
                script_args.malformed_resp_reward
            )
            calculate_and_log_metrics(scatter_log_data)

            # Run PPO step
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)  # noqa
            ppo_trainer.log_stats(stats, batch, rewards)

            if global_iter % script_args.val_freq == 0:
                print("-" * 80)
                print("starting validation")

                response_strings = []
                gt_answers = []
                for _, row in tqdm(validation_set):  # noqa
                    prompt = script_args.prompt_format.format(question=row["question"])
                    response_strings.append(infer_single(model, tokenizer, prompt, script_args.max_pred_tokens))
                    gt_answers.append(row["gt_answer"])

                rewards, scatter_log_data = eval_batch(
                    response_strings,
                    gt_answers,
                    oracle,
                    script_args.score_lambda,
                    script_args.malformed_resp_reward
                )
                calculate_and_log_metrics(scatter_log_data, "val_", True)

            if global_iter % script_args.save_freq == 0:
                try:
                    save_with_timeout(ppo_trainer, os.path.join(save_dir, f"iter_{global_iter}"))
                except TimeoutError:
                    print(f"timeout during saving iter {global_iter}, skip")

            global_iter += 1


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, PPOConfig, ModelConfig))  # noqa
    main(*parser.parse_args_and_config())
