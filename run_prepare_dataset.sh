python prepare_dataset.py \
    --output_csv_path "trivia_qa-rc.wikipedia.nocontext-train.csv" \
    --dataset_path "mandarjoshi/trivia_qa" \
    --dataset_name "rc.wikipedia.nocontext" \
    --dataset_split "train" \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --load_in_4bit
