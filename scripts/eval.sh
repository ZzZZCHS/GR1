source /mnt/yangsizhe/miniconda3/bin/activate /mnt/yangsizhe/miniconda3/envs/gr1
export GIT_PYTHON_REFRESH=quiet
calvin_dataset_path='/mnt/yangsizhe/projects/common/calvin/dataset/task_ABCD_D'
calvin_conf_path='/mnt/yangsizhe/projects/common/calvin/calvin_models/conf'

node=1
node_num=8

resume_from_checkpoint=/mnt/yangsizhe/projects/pretrain_for_manipulation/exp/4090_finetune_calvin_except_lang_1e-3_6query_gr1predimg_epoch19_partial40/23.pth
IFS='/' read -ra path_parts <<< "$resume_from_checkpoint"
run_name="${path_parts[-2]}"
log_name="${path_parts[-1]}"
log_folder="eval_logs/$run_name"
mkdir -p "$log_folder"
log_file="eval_logs/$run_name/evaluate_$log_name.log"
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10081 eval.py \
    --checkpoint_path /mnt/yangsizhe/projects/pretrain_for_manipulation \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
    --calvin_dataset ${calvin_dataset_path} \
    --calvin_conf_path ${calvin_conf_path} \
    --dataset_resampled \
    --workers 16 \
    --lr_scheduler cosine \
    --save_every_iter 50000 \
    --sequence_length 10 \
    --future_steps 3 \
    --commit \
    --num_epochs 20 \
    --seed 42 \
    --batch_size_calvin 56 \
    --precision fp32 \
    --learning_rate 1e-4 \
    --num_resampler_query 6 \
    --run_name test \
    --resume_from_checkpoint ${resume_from_checkpoint} | tee ${log_file}
