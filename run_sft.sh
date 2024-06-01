python sft.py \
    --train_dataset_path "trivia_qa-rc.wikipedia.nocontext-train-balanced10x1000.csv" \
    --val_dataset_path "trivia_qa-rc.wikipedia.nocontext-validation-balanced10x1000.csv" \
    --output_dir "triviaqa-sft-balanced10x1000" \
    --eval_strategy "epoch" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --save_steps 10 \
    --fp16 \
    --optim "adamw_8bit" \
    --report_to "wandb" \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --torch_dtype "float16" \
    --attn_implementation "flash_attention_2" \
    --use_peft \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
    --load_in_4bit
