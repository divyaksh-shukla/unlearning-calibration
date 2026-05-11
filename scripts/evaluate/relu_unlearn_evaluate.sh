#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


models=(
    # "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
    "phi-1_5"
)
per_device_train_batch_size=4 # Effective batch size 32 on two GPUs with gradent_accumulation_steps=8

splits=(
    # "forget10 holdout10 retain90 checkpoint-60"
    # "forget05 holdout05 retain95 checkpoint-30"
    "forget01 holdout01 retain99 checkpoint-6"
)

algos=(
    "grad_ascent locuslab/phi_grad_ascent_1e-05"
    # "grad_diff locuslab/phi_grad_diff_1e-05"
    # "KL_div locuslab/phi_KL_1e-05"
    # "IDK locuslab/phi_idk_1e-05"
)

# ########################################################################################################################
# ########################################### Pretrained models on ReLU ##################################################
# ########################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    checkpoint_split=$(echo $split | cut -d' ' -f4)

    for model in "${models[@]}"; do

        for algo in "${algos[@]}"; do
            algo_name=$(echo $algo | cut -d' ' -f1)
            algo_path=$(echo $algo | cut -d' ' -f2)

            algo_path=${algo_path}_${forget_split}

            echo "Evaluating ${model} with ${algo_name} on split ${split} using path ${algo_path}"
        
            CUDA_VISIBLE_DEVICES=2 python src/eval.py \
            experiment=eval/relu/default.yaml \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            holdout_split=${holdout_split} \
            eval.relu.forget_split=${forget_split} \
            eval.relu.retain_split=${retain_split} \
            eval.relu.holdout_split=${holdout_split} \
            task_name=relu_${model}_${algo_name}_${forget_split}_batch1 \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=${algo_path} \
            model.model_args.revision=${checkpoint_split}
        done
    done
done
