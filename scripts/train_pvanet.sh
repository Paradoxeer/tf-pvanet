set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
model_dir=$2

python train_pvanet.py --train_data_path=cifar10/data_batch* \
  --train_dir=${model_dir} \
  --dataset='cifar10' \
  --max_train_steps=500000 \
  --weight_decay=0.0002 \
  --learning_rate_type='exponential' \
  --learning_rate=0.1 \
  --exponential_decay_rate=0.1 \
  --decay_step=40000 \
  --min_learning_rate=0.0001 \
  --optimizer='mom' \
  --momentum=0.9 \
  --num_gpus=1
