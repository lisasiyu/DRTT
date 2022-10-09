export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
code_dir=
export PYTHONPATH=$code_dir:$PYTHONPATH
model_dir=${code_dir}/checkpoints
data_dir=$code_dir/data/train/chen_data_org
s2t_model=${model_dir}/nist_chen_baseline/checkpoint_best.pt
t2s_model=${model_dir}/nist_ench_baseline/checkpoint_best.pt
mlm_model=${model_dir}/mlm_ch_1e-4/checkpoint_best.pt
tlm_model=${model_dir}/tlm_chen_1e-4/checkpoint_best.pt

s2t_data=${code_dir}/data-bin/chen_baseline
t2s_data=${code_dir}/data-bin/chen_baseline
mlm_data=${code_dir}/data-bin/mlm_ch/
tlm_data=${code_dir}/data-bin/tlm_chen/

s2t_codes=${data_dir}/codes_ch
t2s_codes=${data_dir}/codes_en
mlm_codes=${code_dir}/mlm_ch/codes
tlm_codes=${code_dir}/tlm_chen/codes

src_file_all=$data_dir/train_bpe30k.ch.shuf.delbpe
tgt_file_all=$data_dir/train_bpe30k.en.shuf.delbpe
out_file_all=$code_dir/ch-en_out_k1_rp0.2/
align_file_all=$data_dir/train.ch-en.phrase
stopword_file=${code_dir}/ch_stop_words.npy

if [ ! -d $out_file_all ];then
    mkdir -p $out_file_all
    chmod -R 777 $out_file_all
fi

date

PYTHONIOENCODING=utf-8 python -u ${code_dir}/data_aug.py \
    --s2t_model $s2t_model \
    --t2s_model $t2s_model \
    --mlm_model $mlm_model \
    --tlm_model $tlm_model \
    --s2t_data $s2t_data \
    --t2s_data $t2s_data \
    --mlm_data $mlm_data \
    --tlm_data $tlm_data \
    --s2t_codes $s2t_codes \
    --t2s_codes $t2s_codes \
    --mlm_codes $mlm_codes \
    --tlm_codes $tlm_codes \
    --align_file $align_file_all \
    --stopword_file $stopword_file \
    --src_file $src_file_all \
    --tgt_file $tgt_file_all \
    --out_file $out_file_all/out \
    --k 1 \
    --replace_rate 0.2
date
