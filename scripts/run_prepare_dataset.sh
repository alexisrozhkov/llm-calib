python src/prepare_dataset.py \
    --output_csv_path "trivia_qa-rc.wikipedia.nocontext-train.csv" \
    --dataset_path "mandarjoshi/trivia_qa" \
    --dataset_name "rc.wikipedia.nocontext" \
    --dataset_split "train" \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --load_in_4bit

python src/prepare_dataset.py \
    --output_csv_path "trivia_qa-rc.wikipedia.nocontext-validation.csv" \
    --dataset_path "mandarjoshi/trivia_qa" \
    --dataset_name "rc.wikipedia.nocontext" \
    --dataset_split "validation" \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --load_in_4bit

python src/postprocess_dataset.py \
    --input_csv_path "trivia_qa-rc.wikipedia.nocontext-train.csv" \
    --output_csv_path "trivia_qa-rc.wikipedia.nocontext-train-balanced10x1000.csv" \
    --max_word_count_diff 2 \
    --balance \
    --score_levels 10 \
    --examples_per_level 1000

python src/postprocess_dataset.py \
    --input_csv_path "trivia_qa-rc.wikipedia.nocontext-validation.csv" \
    --output_csv_path "trivia_qa-rc.wikipedia.nocontext-validation-balanced10x100.csv" \
    --max_word_count_diff 2 \
    --balance \
    --score_levels 10 \
    --examples_per_level 100
