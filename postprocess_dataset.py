from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from trl import TrlParser


def normalise_answer(x: str) -> str:
    out = (str(x)
           .replace(".", "")
           .replace(",", "")
           .replace("‘", "")
           .replace("’", "")
           .replace("“", "")
           .replace("”", "")
           .replace("'", "")
           .replace('"', "")).lower().strip()

    out = (out
           .replace("one", "1")
           .replace("two", "2")
           .replace("three", "3")
           .replace("four", "4")
           .replace("five", "5")
           .replace("six", "6")
           .replace("seven", "7")
           .replace("eight", "8")
           .replace("nine", "9")
           .replace("zero", "0"))

    if out.startswith("to "):
        out = out[3:]

    if out.startswith("in "):
        out = out[3:]

    if out.startswith("the "):
        out = out[4:]

    if out.startswith("a "):
        out = out[2:]

    return out


def count_words(s: str) -> int:
    return len(list(filter(lambda x: len(x) > 0, s.split(" "))))


def reeval_numeric_answers(gt_answer: pd.Series, pred_answer: pd.Series) -> np.array:
    return 1 - np.abs(gt_answer - pred_answer) / np.maximum(gt_answer, pred_answer)


@dataclass
class ScriptArguments:
    input_csv_path: str = field(metadata={"help": "Path to the output file from prepare_dataset.py"})
    output_csv_path: str = field(metadata={"help": "Path where to store the postprocessed CSV"})
    max_word_count_diff: Optional[int] = field(default=2, metadata={"help": "Maximum difference in the number of words"
                                                                            " in the answers in the retained examples"})
    balance: Optional[bool] = field(default=False, metadata={"help": "Whether to balance the score values or not"})
    score_levels: Optional[int] = field(default=10, metadata={"help": "Number of confidence buckets for balancing"})
    examples_per_level: Optional[int] = field(default=1000, metadata={"help": "Maximum number of examples per bucket"})


def main(
    input_csv_path: str,
    output_csv_path: str,
    max_word_count_diff: int,
    balance: bool,
    score_levels: int,
    examples_per_level: int,
):
    df = pd.read_csv(input_csv_path)

    # normalise answers
    df["gt_answer_normalised"] = df["gt_answer"].apply(normalise_answer)
    df["pred_answer_normalised"] = df["pred_answer"].apply(normalise_answer)

    # count numbers of words
    df["gt_word_count"] = df["gt_answer_normalised"].apply(count_words)
    df["pred_word_count"] = df["pred_answer_normalised"].apply(count_words)
    df["word_count_diff"] = np.abs(df["gt_word_count"] - df["pred_word_count"])

    # drop examples with difference in number of words > 2 (BLEURT is less reliable in such cases)
    df_filtered = df[~(df["word_count_diff"] > max_word_count_diff)]
    print(f"dropped examples with word num difference > {max_word_count_diff}: {len(df)} -> {len(df_filtered)}")

    # assign perfect score to examples with identical normalised answers
    prev_perfect_score = sum(df_filtered["score"] == 1.0)
    normalised_identical = df_filtered["gt_answer_normalised"] == df_filtered["pred_answer_normalised"]
    df_filtered.loc[normalised_identical, "score"] = 1.0
    print(f"perfect score before / after normalisation: {prev_perfect_score} -> {sum(df_filtered['score'] == 1.0)}")

    # increase score for examples where gt normalised answer is fully contained in pred normalised answer
    normalised_answers_contained = df_filtered.apply(lambda x: x.gt_answer_normalised in x.pred_answer_normalised, axis=1)
    print("gt normalised contained in pred normalised:", sum(normalised_answers_contained))
    df_filtered.loc[normalised_answers_contained, "score"] = (1 + df_filtered.loc[normalised_answers_contained, "score"])/2

    # when both normalised answers are digits only - set score to be abs difference / max value
    both_numeric_answers = df_filtered.apply(lambda x: x.gt_answer_normalised.isnumeric() and x.pred_answer_normalised.isnumeric(), axis=1)

    df_filtered.loc[both_numeric_answers, "score"] = reeval_numeric_answers(
        df_filtered.loc[both_numeric_answers, "gt_answer_normalised"].astype(float),
        df_filtered.loc[both_numeric_answers, "pred_answer_normalised"].astype(float)
    )

    if balance:
        # split examples into bins according to the scaled score, rounded to the closest integer
        df_filtered["score_class"] = (np.round(df_filtered["score"].values * (score_levels - 1))).astype(int)

        balanced_chunks = []
        for score_level in range(score_levels):
            question_subset = df_filtered[df_filtered["score_class"]==score_level]
            sampled = question_subset.sample(examples_per_level, replace=len(question_subset) < examples_per_level)
            balanced_chunks.append(sampled)
            print(score_level, len(question_subset))

        df_balanced = pd.concat(balanced_chunks).sample(frac=1).reset_index(drop=True)

        del df_balanced["score_class"]

        print("duplicates", sum(df_balanced.value_counts("question_id") > 1))

        df_balanced.to_csv(output_csv_path, index=False)

    else:
        df_filtered.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    args: ScriptArguments = TrlParser((ScriptArguments,)).parse_args_and_config()[0]  # noqa

    main(
        args.input_csv_path,
        args.output_csv_path,
        args.max_word_count_diff,
        args.balance,
        args.score_levels,
        args.examples_per_level
    )
