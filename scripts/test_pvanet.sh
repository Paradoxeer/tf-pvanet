set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
all_root=$2

python eval_pvanet.py --eval_data_path=cifar10/test_batch* \
  --eval_dir=${all_root} \
  --checkpoint_path=${all_root} \
  --log_root=${all_root} \
  --dataset='cifar10' \
  --batch_size=100 \
  --eval_batch_count=50 \
  --eval_once=False \
  --weight_decay=0.0002 \
  --num_gpus=1 
