export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
code_dir=
replace_rate=$1
export PYTHONPATH=$code_dir:$PYTHONPATH
out_file=$code_dir/noisy_data_enfr/replace_both/$replace_rate
if [ ! -d $out_file ];then
    echo $out_file
    mkdir -p $out_file
    chmod -R 777 $out_file
fi
PYTHONIOENCODING=utf-8 python $code_dir/noisy_text_ende.py $replace_rate
