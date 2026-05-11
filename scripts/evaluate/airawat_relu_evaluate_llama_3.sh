#! /bin/bash

#SBATCH --job-name=relu_evaluate_llama_3
#SBATCH -N 1
#SBATCH --partition=dgxnp
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out


echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is : $SLURM_SUBMIT_DIR"

cd $SLURM_SUBMIT_DIR

# source /opt/spack/env.sh
# module load python/3.11.9-gcc-11.4.0-aq3ndix
conda activate open_unlearning

http_proxy=http://proxy-10g.10g.siddhi.param:9090
https_proxy=http://proxy-10g.10g.siddhi.param:9090


#######################################################

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


models=(
    # "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)
per_device_train_batch_size=4 # Effective batch size 32 on two GPUs with gradent_accumulation_steps=8

splits=(
    # "forget10 holdout10 retain90"
    # "forget05 holdout05 retain95"
    "forget01 holdout01 retain99"
)

# ########################################################################################################################
# ############################################ Retained models on ReLU ###################################################
# ########################################################################################################################

# for split in "${splits[@]}"; do
#     forget_split=$(echo $split | cut -d' ' -f1)
#     holdout_split=$(echo $split | cut -d' ' -f2)
#     retain_split=$(echo $split | cut -d' ' -f3)
    
#     for model in "${models[@]}"; do

#         echo "Evaluating retained ${model} on forget split ${forget_split}, holdout split ${holdout_split}, retain split ${retain_split}"
    
#         CUDA_VISIBLE_DEVICES=0 python src/eval.py \
#         experiment=eval/relu/default.yaml \
#         forget_split=${forget_split} \
#         retain_split=${retain_split} \
#         holdout_split=${holdout_split} \
#         eval.relu.forget_split=${forget_split} \
#         eval.relu.retain_split=${retain_split} \
#         eval.relu.holdout_split=${holdout_split} \
#         task_name=relu_${model}_${retain_split}_nochat \
#         model=${model} \
#         model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_${retain_split}
#     done
# done

# ########################################################################################################################
# ########################################### Pretrained models on ReLU ##################################################
# ########################################################################################################################

# for split in "${splits[@]}"; do
#     forget_split=$(echo $split | cut -d' ' -f1)
#     holdout_split=$(echo $split | cut -d' ' -f2)
#     retain_split=$(echo $split | cut -d' ' -f3)
    
#     for model in "${models[@]}"; do

#         echo "Evaluating Pretrained ${model} on forget split ${forget_split}, holdout split ${holdout_split}, retain split ${retain_split}"
    
#         CUDA_VISIBLE_DEVICES=0 python src/eval.py \
#         experiment=eval/relu/default.yaml \
#         forget_split=${forget_split} \
#         retain_split=${retain_split} \
#         holdout_split=${holdout_split} \
#         eval.relu.forget_split=${forget_split} \
#         eval.relu.retain_split=${retain_split} \
#         eval.relu.holdout_split=${holdout_split} \
#         task_name=relu_${model}_pretrained_${retain_split}_nochat \
#         model=${model} \
#         retain_logs_path=saves/eval/relu_${model}_${retain_split}_nochat/RELU_EVAL.json \
#         model.model_args.pretrained_model_name_or_path=meta-llama/${model}
#     done
# done

########################################################################################################################
######################################### Full Finetuned models on ReLU ################################################
########################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    
    for model in "${models[@]}"; do

        echo "Evaluating full-finetuned ${model} on forget split ${forget_split}, holdout split ${holdout_split}, retain split ${retain_split}"
    
        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/relu/default.yaml \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        holdout_split=${holdout_split} \
        eval.relu.forget_split=${forget_split} \
        eval.relu.retain_split=${retain_split} \
        eval.relu.holdout_split=${holdout_split} \
        task_name=relu_${model}_full_${retain_split}_nochat \
        model=${model} \
        retain_logs_path=saves/eval/relu_${model}_${retain_split}_nochat/RELU_EVAL.json \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full
        # model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full
        # model.model_args.pretrained_model_name_or_path=saves/finetune/relu_${model}_${retain_split}
    done
done




########################################################################################################################
########################################### RETAIN Finetuned RELU ######################################################
########################################################################################################################

# for split in "${splits[@]}"; do
#     forget_split=$(echo $split | cut -d' ' -f1)
#     holdout_split=$(echo $split | cut -d' ' -f2)
#     retain_split=$(echo $split | cut -d' ' -f3)
    
#     for model in "${models[@]}"; do
#         CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#         src/train.py experiment=finetune/tofu/default.yaml \
#         task_name=tofu_${model}_${retain_split} \
#         model=${model} \
#         data/datasets@data.train=TOFU_QA_retain \
#         data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
#         trainer.args.per_device_train_batch_size=4 \
#         trainer.args.ddp_find_unused_parameters=true \
#         trainer.args.gradient_checkpointing=true

    
#         CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
#         forget_split=${forget_split} \
#         holdout_split=${holdout_split} \
#         task_name=tofu_${model}_${retain_split} \
#         model=${model} \
#         model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_${retain_split}
#     done
# done


# ########################################################################################################################
# ########################################### FULL Finetuned RELU models #################################################
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

