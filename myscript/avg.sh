export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
code_dir=
task=$1
checkpoint_dir=${code_dir}/checkpoints/$task
export PYTHONPATH=$code_dir:$PYTHONPATH
python $code_dir/scripts/average_checkpoints.py --inputs ${checkpoint_dir} --num-update-checkpoints 5 --output ${checkpoint_dir}/avg.pt

