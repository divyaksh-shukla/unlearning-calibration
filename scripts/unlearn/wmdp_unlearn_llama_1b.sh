#!/bin/bash


export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

export HF_HOME="/DATA3/divyaksh/.cache/huggingface/hub"

models=(
    "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
)
trainers_experiments=(
    "GradAscent unlearn/wmdp/default.yaml"
    "GradDiff unlearn/wmdp/default.yaml"
    "NPO unlearn/wmdp/default.yaml"
    "DPO unlearn/wmdp/default.yaml"
    "RMU  unlearn/wmdp/default.yaml"
    "GradDiffKLUniform  unlearn/wmdp/default.yaml"
    "PDU  unlearn/wmdp/default.yaml"
    "SatImp  unlearn/wmdp/default.yaml"
    "SimNPO  unlearn/wmdp/default.yaml"
    "UNDIAL  unlearn/wmdp/default.yaml"
    "WGA  unlearn/wmdp/default.yaml"
    "CEU  unlearn/wmdp/default.yaml"
)
# splits=(
#     "forget01 holdout01 retain99"
#     "forget05 holdout05 retain95"
#     "forget10 holdout10 retain90"
# )


per_device_train_batch_size=32 
gradient_accumulation_steps=1 # on one gpus would make effective batch size 32


########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################


for model in "${models[@]}"; do
    for trainer_experiment in "${trainers_experiments[@]}"; do
        trainer=$(echo $trainer_experiment | cut -d' ' -f1)
        experiment=$(echo $trainer_experiment | cut -d' ' -f2)
        
        task_name=wmdp_${model}_${forget_split}_${trainer} 
        echo ${task_name}: Unlearning ${model_path} using ${trainer}

        # Unlearn
        accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
        src/train.py --config-name=unlearn.yaml \
        experiment=${experiment} \
        trainer=${trainer} \
        task_name=${task_name} \
        model=${model} \
        trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
        trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true 

        # # Eval
        # CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        # experiment=eval/tofu/default.yaml \
        # forget_split=${forget_split} \
        # holdout_split=${holdout_split} \
        # model=${model} \
        # task_name=${task_name} \
        # model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
        # paths.output_dir=saves/unlearn/${task_name}/evals \
        # retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
    done
done