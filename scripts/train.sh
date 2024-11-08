set -x
export MASTER_PORT=$((54000 + $RANDOM % 10000))
# calvin_dataset_path='/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/GR1/data/calvin_debug_dataset'
data_dir='/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_data'

node=1
node_num=8

exp_name=robomimic_train_addmask
batch_size=8
lr=5e-4
weight_decay=1e-2
epochs=5
transformer_layers=6
transformer_hidden_dim=512
transformer_heads=8
num_resampler_query=9
addmask=True

run_name="$(date +"%Y%m%d_%H%M%S")"_"$exp_name"_bs"$batch_size"_lr"$lr"_ep"$epochs"_decay"$weight_decay"_layers"$transformer_layers"_dim"$transformer_hidden_dim"_heads"$transformer_heads"_samplernum"$num_resampler_query"_addmask"$addmask"

which python
which torchrun

torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=${MASTER_PORT} train.py \
    --checkpoint_path ./pretrain/ \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --bf16_module "vision_encoder" \
    --dataset_resampled \
    --workers 24 \
    --lr_scheduler cosine \
    --save_every_iter 500000 \
    --sequence_length 10 \
    --future_steps 1 \
    --commit \
    --num_epochs $epochs \
    --seed 42 \
    --gradient_accumulation_steps 4 \
    --batch_size_calvin $batch_size \
    --precision fp32 \
    --learning_rate $lr \
    --weight_decay $weight_decay \
    --num_resampler_query $num_resampler_query \
    --run_name "$run_name" \
    --transformer_layers $transformer_layers \
    --transformer_hidden_dim $transformer_hidden_dim \
    --transformer_heads $transformer_heads \
    --save_checkpoint \
    --config "configs/addmask.json" \
    --addmask $addmask \
    --report_to_wandb
    # --data_dir "$data_dir"
    # --calvin_dataset "$calvin_dataset_path" \

