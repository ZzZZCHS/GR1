export GIT_PYTHON_REFRESH=quiet
# calvin_dataset_path='/mnt/hwfile/OpenRobotLab/robomani/calvin_data/task_ABCD_D'
# calvin_conf_path='/mnt/hwfile/OpenRobotLab/huanghaifeng/GR1/calvin/calvin_models/conf'

node=1
node_num=1

resume_from_checkpoint=/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/GR1/pretrain/exp/20241106_133343_robomimic_train_1_noimageloss_bs1_lr2e-4_steps1000_decay0.0/999.pth
IFS='/' read -ra path_parts <<< "$resume_from_checkpoint"
run_name="${path_parts[-2]}"
log_name="${path_parts[-1]}"
log_folder="eval_logs/$run_name"
mkdir -p "$log_folder"
log_file="eval_logs/$run_name/evaluate_$log_name.log"
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10082 eval_robocasa.py \
    --checkpoint_path ./pretrain \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
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
    --run_name ep999 \
    --config "configs/noadd.json" \
    --val_domain train \
    --resume_from_checkpoint ${resume_from_checkpoint} | tee ${log_file}
    # --calvin_dataset ${calvin_dataset_path} \
    # --calvin_conf_path ${calvin_conf_path} \

