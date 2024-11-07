export GIT_PYTHON_REFRESH=quiet
# calvin_dataset_path='/mnt/hwfile/OpenRobotLab/robomani/calvin_data/task_ABCD_D'
# calvin_conf_path='/mnt/hwfile/OpenRobotLab/huanghaifeng/GR1/calvin/calvin_models/conf'

node=1
node_num=8

lr=5e-4
weight_decay=1e-3
transformer_layers=6
transformer_hidden_dim=512
transformer_heads=8
num_resampler_query=9

resume_from_checkpoint=/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/GR1/pretrain/exp/20241107_032924_robomimic_train_countertocab_bs8_lr5e-4_ep20_decay1e-3_layers6_dim512_heads8_samplernum9/5.pth
IFS='/' read -ra path_parts <<< "$resume_from_checkpoint"
run_name="${path_parts[-2]}"
log_name="${path_parts[-1]}"
log_folder="eval_logs/$run_name"
mkdir -p "$log_folder"
log_file="eval_logs/$run_name/evaluate_$log_name.log"
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10081 eval_robocasa.py \
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
    --future_steps 1 \
    --commit \
    --num_epochs 20 \
    --seed 42 \
    --batch_size_calvin 56 \
    --precision fp32 \
    --learning_rate $lr \
    --weight_decay $weight_decay \
    --num_resampler_query $num_resampler_query \
    --transformer_layers $transformer_layers \
    --transformer_hidden_dim $transformer_hidden_dim \
    --transformer_heads $transformer_heads \
    --run_name ep5_val_friction1 \
    --config "configs/noadd.json" \
    --val_domain val \
    --resume_from_checkpoint ${resume_from_checkpoint} | tee ${log_file}
    # --calvin_dataset ${calvin_dataset_path} \
    # --calvin_conf_path ${calvin_conf_path} \

