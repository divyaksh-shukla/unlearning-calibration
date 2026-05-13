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
    "forget10 holdout10 retain90"
    # "forget05 holdout05 retain95"
    # "forget01 holdout01 retain99"
)

export CUDA_VISIBLE_DEVICES=1



########################################################################################################################
########################################### RETAIN Finetuned TOFU ######################################################
########################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    
    for model in "${models[@]}"; do
    
        python src/eval.py experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        holdout_split=${holdout_split} \
        task_name=tofu_${model}_${retain_split}_gpu_test \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=locuslab/tofu_ft_${retain_split}_phi-1.5
    done
done


########################################################################################################################
############################################## PRETRAINED TOFU #########################################################
########################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    
    for model in "${models[@]}"; do
    
        python src/eval.py experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        holdout_split=${holdout_split} \
        task_name=tofu_${model}_pretrained_${retain_split}_gpu_test \
        model=${model} \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}_gpu_test/TOFU_EVAL.json \
        model.model_args.pretrained_model_name_or_path=microsoft/phi-1_5
    done
done


########################################################################################################################
############################################ FULL Finetuned TOFU #######################################################
########################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    
    for model in "${models[@]}"; do
    
        python src/eval.py experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        holdout_split=${holdout_split} \
        task_name=tofu_${model}_ft_${retain_split}_gpu_test \
        model=${model} \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}_gpu_test/TOFU_EVAL.json \
        model.model_args.pretrained_model_name_or_path=locuslab/tofu_ft_phi-1.5
    done
done


# ########################################################################################################################
# ########################################### FULL Finetuned TOFU models #################################################
# ########################################################################################################################


# for model in "${models[@]}"; do
#     CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#     src/train.py experiment=finetune/tofu/default.yaml \
#     task_name=tofu_${model}_full \
#     model=${model} \
#     data/datasets@data.train=TOFU_QA_full \
#     data.train.TOFU_QA_full.args.hf_args.name=full \
#     trainer.args.per_device_train_batch_size=4 \
#     trainer.args.ddp_find_unused_parameters=true \
#     trainer.args.gradient_checkpointing=true

#     # Evaluate the full models on each forget split
#     for split in "${splits[@]}"; do
#         forget_split=$(echo $split | cut -d' ' -f1)
#         holdout_split=$(echo $split | cut -d' ' -f2)
#         retain_split=$(echo $split | cut -d' ' -f3)

#         CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
#         forget_split=${forget_split} \
#         holdout_split=${holdout_split} \
#         task_name=tofu_${model}_full_${forget_split} \
#         model=${model} \
#         model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_full \
#         retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
#         paths.output_dir=saves/eval/tofu_${model}_full/evals_${forget_split}
#     done
# done