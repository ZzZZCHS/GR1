set -x
calvin_dataset_path='/mnt/hwfile/OpenRobotLab/robomani/calvin_data/task_ABCD_D'

node=1
node_num=1

which python
which torchrun

torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10079 train.py \
    --checkpoint_path ./pretrain/ \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --bf16_module "vision_encoder" \
    --calvin_dataset ${calvin_dataset_path} \
    --dataset_resampled \
    --workers 16 \
    --lr_scheduler cosine \
    --save_every_iter 50000 \
    --sequence_length 10 \
    --future_steps 3 \
    --commit \
    --num_epochs 20 \
    --seed 42 \
    --gradient_accumulation_steps 4 \
    --batch_size_calvin 16 \
    --precision fp32 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name gr1_train \
    --save_checkpoint \
    --report_to_wandb 
