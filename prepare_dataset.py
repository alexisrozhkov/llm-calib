from dataclasses import field, dataclass
from typing import Optional

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig
from trl.commands.cli_utils import TrlParser


@dataclass
class ScriptArguments:
    dataset_path: str = field(metadata={"help": "Path to a dataset to use"})
    dataset_name: str = field(metadata={"help": "Name of the dataset subset to use"})
    dataset_split: str = field(metadata={"help": "Name of the dataset split to use"})
    output_csv_path: str = field(metadata={"help": "Path where to write resulting CSV"})

    save_freq: Optional[int] = field(default=100, metadata={"help": "Make sure we don't lose progress if something fails midway"})
    prompt_format: Optional[str] = field(default="Reply in a few words.\nQ: {question}\nA:", metadata={"help": "Q/A prompt format"})
    max_pred_tokens: Optional[int] = field(default=16, metadata={"help": "Maximum number of tokens to predict"})
    oracle_model: Optional[str] = field(default="BLEURT-20", metadata={"help": "Type of BLEURT model to use"})


def main(script_args: ScriptArguments, model_config: ModelConfig):
    device = "cuda"

    oracle = evaluate.load("bleurt", config_name=script_args.oracle_model)

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        load_in_4bit=model_config.load_in_4bit,
        torch_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.float16,
        device_map={"": device},
        attn_implementation=model_config.attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(script_args.dataset_path, script_args.dataset_name, split=script_args.dataset_split)

    generation_kwargs = {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": script_args.max_pred_tokens,
    }

    dataset_out = []

    for example in tqdm(dataset):
        question_id = example["question_id"]
        question = example["question"]
        gt_answer = example["answer"]["value"]

        inputs = tokenizer([script_args.prompt_format.format(question=question),], return_tensors="pt").to(device)
        prompt_length = inputs['input_ids'].shape[1]

        outputs = model.generate(**inputs, **generation_kwargs)
        output_str = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

        pred_answer = output_str.split("\n")[0].strip()
        if pred_answer[-1] == ".":
            pred_answer = pred_answer[:-1]

        if pred_answer == gt_answer:
            score = 1.0
        else:
            score_raw = oracle.compute(predictions=[pred_answer], references=[gt_answer])["scores"][0]
            score = np.clip(score_raw, 0, 1)

            if gt_answer.lower() in pred_answer.lower():
                score = (score + 1) / 2

        print("-"*80)
        print(question)
        print(gt_answer)
        print(pred_answer)
        print(score)

        labeled_example = {
            "question_id": question_id,
            "question": question,
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
            "score": score
        }

        dataset_out.append(labeled_example)

        if len(dataset_out) % script_args.save_freq == 0:
            df = pd.DataFrame(dataset_out)
            df.to_csv(script_args.output_csv_path, index=False)

    df = pd.DataFrame(dataset_out)
    df.to_csv(script_args.output_csv_path, index=False)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, ModelConfig))  # noqa
    main(*parser.parse_args_and_config())
