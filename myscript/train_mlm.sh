export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7
root_dir=
code_dir=
python=
fastalign_tool=${root_dir}/fast_align/build
data_dir=${code_dir}/data/train/chen_data_org
out_dir=${code_dir}/mlm_ch

if [ ! -d $out_dir ];then
    mkdir -p $out_dir
    chmod -R 777 $out_dir
fi

src=ch
tgt=en

train_src_file=${data_dir}/train_bpe30k.ch.shuf.delbpe
valid_src_file=${data_dir}/nist_valid06_test02-08/mt06_src_u8.BPE.delbpe
test_src_file=${data_dir}/nist_valid06_test02-08/mt02_src_u8.BPE.delbpe

echo "applying mask"
#creat_mask.py --src_file  --file_out --file_label
${python} ${code_dir}/creat_mask.py ${train_src_file} ${out_dir}/train.mask.${src} ${out_dir}/train.label.${tgt} 0.5
${python} ${code_dir}/creat_mask.py ${valid_src_file} ${out_dir}/valid.mask.${src} ${out_dir}/valid.label.${tgt} 0.3
${python} ${code_dir}/creat_mask.py ${valid_src_file} ${out_dir}/test.mask.${src} ${out_dir}/test.label.${tgt} 0.1

#: << 'comment'
echo "applying bpe"
BPEROOT=${root_dir}/subword-nmt/subword_nmt
BPE_TOKENS=30000
SCRIPTS=${root_dir}/mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPE_CODE=${out_dir}/codes

cat ${out_dir}/train.mask.${src}  ${out_dir}/train.label.${tgt} > ${out_dir}/tmp
echo "learn bpe"
$python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < ${out_dir}/tmp > $BPE_CODE
echo "apply bpe"

$python $BPEROOT/apply_bpe.py -c $BPE_CODE < ${out_dir}/train.mask.${src} > ${out_dir}/train.${src}
$python $BPEROOT/apply_bpe.py -c $BPE_CODE < ${out_dir}/train.label.${tgt}  > ${out_dir}/train.${tgt}

$python $BPEROOT/apply_bpe.py -c $BPE_CODE < ${out_dir}/valid.mask.${src} > ${out_dir}/valid.${src}
$python $BPEROOT/apply_bpe.py -c $BPE_CODE < ${out_dir}/valid.label.${tgt}  > ${out_dir}/valid.${tgt}

$python $BPEROOT/apply_bpe.py -c $BPE_CODE < ${out_dir}/test.mask.${src} > ${out_dir}/test.${src}
$python $BPEROOT/apply_bpe.py -c $BPE_CODE < ${out_dir}/test.label.${tgt}  > ${out_dir}/test.${tgt}

echo "shuffle"
${python} ${code_dir}/shuffle_corpus.py --corpus ${out_dir}/train.${src} ${out_dir}/train.${tgt}
${python} ${code_dir}/shuffle_corpus.py --corpus ${out_dir}/valid.${src} ${out_dir}/valid.${tgt}
${python} ${code_dir}/shuffle_corpus.py --corpus ${out_dir}/test.${src} ${out_dir}/test.${tgt}

mkdir ${out_dir}/final
mv ${out_dir}/train.${src}.shuf ${out_dir}/final/train.${src}
mv ${out_dir}/train.${tgt}.shuf ${out_dir}/final/train.${tgt}
mv ${out_dir}/valid.${src}.shuf ${out_dir}/final/valid.${src}
mv ${out_dir}/valid.${tgt}.shuf ${out_dir}/final/valid.${tgt}
mv ${out_dir}/test.${src}.shuf ${out_dir}/final/test.${src}
mv ${out_dir}/test.${tgt}.shuf ${out_dir}/final/test.${tgt}

echo 'start preprocessing'
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7

$python ${code_dir}/fairseq_cli/preprocess.py --source-lang ch --target-lang en\
 --trainpref ${out_dir}/final/train \
 --validpref ${out_dir}/final/valid \
 --testpref ${out_dir}/final/test \
 --destdir ${code_dir}/data-bin/mlm_ch \
 --workers 100

echo 'start training'
learning_rate=1e-4
update=10
checkpoint_dir=${code_dir}/checkpoints/mlm_ch_${learning_rate}
export PYTHONPATH=$code_dir:$PYTHONPATH
if [ ! -d $checkpoint_dir ];then
    mkdir -p $checkpoint_dir
    chmod -R 777 $checkpoint_dir
fi
data_path=$code_dir/data-bin/mlm_ch

PYTHONIOENCODING=utf-8 $python ${code_dir}/fairseq_cli/train.py ${data_path} \
    --task translation \
    --fp16 --ddp-backend=no_c10d \
    --source-lang ch --target-lang en \
    --arch transformer \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0   --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 --warmup-updates 4000  \
    --lr $learning_rate \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --weight-decay 0.0 --max-tokens  4096 \
    --dropout 0.1 \
    --save-dir $checkpoint_dir \
    --update-freq $update \
    --log-interval 100 --save-interval-updates 2000 \
    --max-update 100000 \
    --eval-bleu-print-samples \


