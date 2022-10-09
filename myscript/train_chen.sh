export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
code_dir=
python=
data_dir=$code_dir/ch-en_out_k1_rp0.2/filter_0.01_0.5
$python ${code_dir}/fairseq_cli/preprocess.py --source-lang ch --target-lang en\
 --trainpref $data_dir/train \
 --validpref $data_dir/valid \
 --testpref $data_dir/test \
 --destdir ${code_dir}/data-bin/chen_adv_0.01_0.5 \
 --workers 200


lr=5e-4
dropout=0.1
data_bin=${code_dir}/data-bin/chen_adv_0.01_0.5
update=1

checkpoint_dir=$code_dir/checkpoints/nist_${data_bin}_lr${lr}_drop${dropout}_up${update}
results_path=$code_dir/results/nist_${data_bin}_lr${lr}_drop${dropout}

export PYTHONPATH=$code_dir:$PYTHONPATH
if [ ! -d $checkpoint_dir ];then
    mkdir -p $checkpoint_dir
    chmod -R 777 $checkpoint_dir
fi    

data_path=$code_dir/data-bin/$data_bin
PYTHONIOENCODING=utf-8 /apdcephfs/share_47076/lisalai/anaconda3/bin/python $code_dir/fairseq_cli/train.py ${data_path} \
    --fp16 --ddp-backend=no_c10d \
    --source-lang ch --target-lang en \
    --share-decoder-input-output-embed \
    --arch transformer \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0   --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 --warmup-updates 4000  \
    --lr $lr --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --weight-decay 0.0 --max-tokens  4096 \
    --dropout $dropout \
    --save-dir $checkpoint_dir \
    --update-freq $update \
    --log-interval 100 --save-interval-updates 2000 \
    --max-update 100000 \
    --eval-bleu \
    --eval-tokenized-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \

