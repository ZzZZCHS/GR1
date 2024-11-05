set -x
# calvin_dataset_path='/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/GR1/data/calvin_debug_dataset'
data_dir='/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_1024'

node=1
node_num=8

exp_name=robomimic_train_1_door
batch_size=4
lr=1e-3
weight_decay=0.01
epochs=500

run_name="$(date +"%Y%m%d_%H%M%S")"_"$exp_name"_bs"$batch_size"_lr"$lr"_steps"$epochs"_decay"$weight_decay"

which python
which torchrun

torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10079 train.py \
    --checkpoint_path ./pretrain/ \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --bf16_module "vision_encoder" \
    --dataset_resampled \
    --workers 16 \
    --lr_scheduler cosine \
    --save_every_iter 50000 \
    --sequence_length 10 \
    --future_steps 3 \
    --commit \
    --num_epochs $epochs \
    --seed 42 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin $batch_size \
    --precision fp32 \
    --learning_rate $lr \
    --weight_decay $weight_decay \
    --num_resampler_query 6 \
    --run_name "$run_name" \
    --save_checkpoint \
    --config "configs/noadd_opendoor.json" \
    --delete_previous_checkpoint
    # --report_to_wandb 
    # --data_dir "$data_dir"
    # --calvin_dataset "$calvin_dataset_path" \

