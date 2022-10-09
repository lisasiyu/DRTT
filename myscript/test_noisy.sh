code_dir=
data_dir=
ckpt=
data_bin=
task=
best_checkpoints_dir=
results_path=$code_dir/results/$ckpt
eval_path=
detokenizer=
BPE_CODE_EN=$data_dir/codes_en
BPE_CODE_CH=$code_dir/$task/codes_ch
BPEROOT=

export PYTHONPATH=$code_dir:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
 
sample_array=(0 0.1 0.2 0.3 0.4 0.5)
sample_array1=(0.1 0.2 0.3 0.4 0.5)
echo $ckpt
#:<<'comment'
echo 'replace_both'
data_path_tgt=$code_dir/data/noisy_test/noisy_data_chen/replace_both
data_path_src=$code_dir/$task/noisy/replace_both

if [ ! -d $results_path ];then
    mkdir -p $results_path
    chmod -R 777 $results_path
fi

index=0
for i in ${sample_array[@]};        
do
{
    for j in ${sample_array[*]}
    do
    if [[ $i == $j ]]
        then
        a=$index
    fi
        index=$(( $index + 1 ))
    done
    
    export CUDA_VISIBLE_DEVICES=$a
    #echo $a
    PYTHONIOENCODING=utf-8 python -u ${code_dir}/fairseq_cli/interactive.py \
    ${code_dir}/data-bin/$data_bin \
    --input ${data_path_src}/$i/valid06.bpe.ch \
    --path ${best_checkpoints_dir} \
    --batch-size 256 \
    --buffer-size 256 \
    --beam 5 \
	  --remove-bpe |
    grep "^H-"  | cut -f 3 > ${results_path}/${i}.interactive.out
    $detokenizer/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en <${results_path}/${i}.interactive.out> ${results_path}/${i}.interactive.out.detok

    #$echo 'back translation'
    python $BPEROOT/apply_bpe.py -c $BPE_CODE_EN < ${results_path}/${i}.interactive.out > ${results_path}/${i}.interactive.bpe.out

    PYTHONIOENCODING=utf-8 python -u ${code_dir}/fairseq_cli/interactive.py \
    ${code_dir}/data-bin/nist-en-ch-finetune \
    --input ${results_path}/${i}.interactive.bpe.out \
    --path $code_dir/checkpoints/nist_ench_baseline_ft/checkpoint_best.pt \
    --batch-size 256 \
    --buffer-size 256 \
    --beam 5 \
    --remove-bpe |
    grep "^H-"  | cut -f 3 > ${results_path}/${i}.interactive.back.out

    sed 's/[[:space:]]//g' ${results_path}/${i}.interactive.back.out > ${results_path}/${i}.interactive.back.out.detok
} &
done 
wait

#:<<'comment'
echo 'delete'
data_path_tgt=$code_dir/data/noisy_test/noisy_data_chen/delete
data_path_src=$code_dir/$task/noisy/delete

if [ ! -d $results_path ];then
    mkdir $results_path
    chmod -R 777 $results_path
fi

index=0
for i in ${sample_array[@]};
do
{
    for j in ${sample_array[*]}
    do
    if [[ $i == $j ]]
        then
        a=$index
    fi
        index=$(( $index + 1 ))
    done

    export CUDA_VISIBLE_DEVICES=$a
    PYTHONIOENCODING=utf-8 python -u ${code_dir}/fairseq_cli/interactive.py \
    ${code_dir}/data-bin/$data_bin \
    --input ${data_path_src}/$i/valid06.bpe.ch \
    --path ${best_checkpoints_dir} \
    --batch-size 256 \
    --buffer-size 256 \
    --beam 5 \
	--remove-bpe |
    grep "^H-"  | cut -f 3 > ${results_path}/${i}.del.interactive.out
    $detokenizer/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en <${results_path}/${i}.del.interactive.out> ${results_path}/${i}.del.interactive.out.detok

    #echo 'back translation'
    python $BPEROOT/apply_bpe.py -c $BPE_CODE_EN < ${results_path}/${i}.del.interactive.out > ${results_path}/${i}.del.interactive.bpe.out

    PYTHONIOENCODING=utf-8 python -u ${code_dir}/fairseq_cli/interactive.py \
    ${code_dir}/data-bin/nist-en-ch-finetune \
    --input ${results_path}/${i}.del.interactive.bpe.out \
    --path $code_dir/checkpoints/nist_ench_baseline_ft/checkpoint_best.pt \
    --batch-size 256 \
    --buffer-size 256 \
    --beam 5 \
    --remove-bpe |
    grep "^H-"  | cut -f 3 > ${results_path}/${i}.del.interactive.back.out
    sed 's/[[:space:]]//g' ${results_path}/${i}.del.interactive.back.out>${results_path}/${i}.del.interactive.back.out.detok
} &
done
wait

echo 'swap'
data_path_tgt=$code_dir/data/noisy_test/noisy_data_chen/swap
data_path_src=$code_dir/$task/noisy/swap

if [ ! -d $results_path ];then
    mkdir $results_path
    chmod -R 777 $results_path
fi
index=0
for i in ${sample_array[@]};
do
{
    for j in ${sample_array[*]}
    do
    if [[ $i == $j ]]
        then
        a=$index
    fi
        index=$(( $index + 1 ))
    done

    export CUDA_VISIBLE_DEVICES=$a    
    PYTHONIOENCODING=utf-8 python -u ${code_dir}/fairseq_cli/interactive.py \
    ${code_dir}/data-bin/$data_bin \
    --input ${data_path_src}/$i/valid06.bpe.ch \
    --path ${best_checkpoints_dir} \
    --batch-size 256 \
    --buffer-size 256 \
    --beam 5 \
	--remove-bpe |
    grep "^H-"  | cut -f 3 > ${results_path}/${i}.swap.interactive.out
    $detokenizer/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en <${results_path}/${i}.swap.interactive.out> ${results_path}/${i}.swap.interactive.out.detok
    
    #echo 'back translation'
    python $BPEROOT/apply_bpe.py -c $BPE_CODE_EN < ${results_path}/${i}.swap.interactive.out > ${results_path}/${i}.swap.interactive.bpe.out

    PYTHONIOENCODING=utf-8 python -u ${code_dir}/fairseq_cli/interactive.py \
    ${code_dir}/data-bin/nist-en-ch-finetune \
    --input ${results_path}/${i}.swap.interactive.bpe.out \
    --path $code_dir/checkpoints/nist_ench_baseline_ft/checkpoint_best.pt \
    --batch-size 256 \
    --buffer-size 256 \
    --beam 5 \
    --remove-bpe |
    grep "^H-"  | cut -f 3 > ${results_path}/${i}.swap.interactive.back.out
    sed 's/[[:space:]]//g' ${results_path}/${i}.swap.interactive.back.out>${results_path}/${i}.swap.interactive.back.out.detok

} &
done
wait
#comment
echo 'replace_src'
data_path_tgt=$code_dir/data/noisy_test/noisy_data_chen/replace_src
data_path_src=$code_dir/$task/noisy/replace_src

if [ ! -d $results_path ];then
    mkdir $results_path
    chmod -R 777 $results_path
fi
index=0
for i in ${sample_array[@]};
do
{
    for j in ${sample_array[*]}
    do
    if [[ $i == $j ]]
        then
        a=$index
    fi
        index=$(( $index + 1 ))
    done

    export CUDA_VISIBLE_DEVICES=$a
    PYTHONIOENCODING=utf-8 python -u ${code_dir}/fairseq_cli/interactive.py \
    ${code_dir}/data-bin/$data_bin \
    --input ${data_path_src}/$i/valid06.bpe.ch \
    --path ${best_checkpoints_dir} \
    --batch-size 256 \
    --buffer-size 256 \
    --beam 5 \
    --remove-bpe |
    grep "^H-"  | cut -f 3 > ${results_path}/${i}.rep.interactive.out
    $detokenizer/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en <${results_path}/${i}.rep.interactive.out> ${results_path}/${i}.rep.interactive.out.detok

   # echo 'back translation'
    python $BPEROOT/apply_bpe.py -c $BPE_CODE_EN < ${results_path}/${i}.rep.interactive.out > ${results_path}/${i}.rep.interactive.bpe.out

    PYTHONIOENCODING=utf-8 python -u ${code_dir}/fairseq_cli/interactive.py \
    ${code_dir}/data-bin/nist-en-ch-finetune \
    --input ${results_path}/${i}.rep.interactive.bpe.out \
    --path $code_dir/checkpoints/nist_ench_baseline_ft/checkpoint_best.pt \
    --batch-size 256 \
    --buffer-size 256 \
    --beam 5 \
    --remove-bpe |
    grep "^H-"  | cut -f 3 > ${results_path}/${i}.rep.interactive.back.out
    sed 's/[[:space:]]//g' ${results_path}/${i}.rep.interactive.back.out>${results_path}/${i}.rep.interactive.back.out.detok
}&
done
wait

echo 'insert'
data_path_tgt=$code_dir/data/noisy_test/noisy_data_chen/insert
data_path_src=$code_dir/$task/noisy/insert

if [ ! -d $results_path ];then
    mkdir $results_path
    chmod -R 777 $results_path
fi
index=0
for i in ${sample_array[@]};
do
{
    for j in ${sample_array[*]}
    do
    if [[ $i == $j ]]
        then
        a=$index
    fi
        index=$(( $index + 1 ))
    done

    export CUDA_VISIBLE_DEVICES=$a
    PYTHONIOENCODING=utf-8 python -u ${code_dir}/fairseq_cli/interactive.py \
    ${code_dir}/data-bin/$data_bin \
    --input ${data_path_src}/$i/valid06.bpe.ch \
    --path ${best_checkpoints_dir} \
    --batch-size 256 \
    --buffer-size 256 \
    --beam 5 \
    --remove-bpe |
    grep "^H-"  | cut -f 3 > ${results_path}/${i}.ins.interactive.out
    $detokenizer/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en <${results_path}/${i}.ins.interactive.out> ${results_path}/${i}.ins.interactive.out.detok


    #echo 'back translation'
    python $BPEROOT/apply_bpe.py -c $BPE_CODE_EN < ${results_path}/${i}.rep.interactive.out > ${results_path}/${i}.ins.interactive.bpe.out

    PYTHONIOENCODING=utf-8 python -u ${code_dir}/fairseq_cli/interactive.py \
    ${code_dir}/data-bin/nist-en-ch-finetune \
    --input ${results_path}/${i}.ins.interactive.bpe.out \
    --path $code_dir/checkpoints/nist_ench_baseline_ft/checkpoint_best.pt \
    --batch-size 256 \
    --buffer-size 256 \
    --beam 5 \
    --remove-bpe |
    grep "^H-"  | cut -f 3 > ${results_path}/${i}.ins.interactive.back.out
    sed 's/[[:space:]]//g' ${results_path}/${i}.ins.interactive.back.out>${results_path}/${i}.ins.interactive.back.out.detok

} &
done
wait

for i in ${sample_array[@]};
do
{
    echo '############################'
    echo $i
    echo 'replace both'
    data_path_tgt=$code_dir/data/noisy_test/noisy_data_chen/replace_both
    data_path_src=$code_dir/$task/noisy/replace_both
    ${eval_path}/multi-bleu.perl -lc  ${data_path_tgt}/${i}/valid06.en <${results_path}/${i}.interactive.out
    sacrebleu ${data_path_tgt}/${i}/valid06.ch.detok -i ${results_path}/${i}.interactive.back.out.detok --tokenize zh -b -w 2

    echo 'delete'
    data_path_tgt=$code_dir/data/noisy_test/noisy_data_chen/delete
    data_path_src=$code_dir/$task/noisy/delete
    ${eval_path}/multi-bleu.perl -lc nist_valid06_test02-08/mt06_trg_u8.tok.sb  <${results_path}/${i}.del.interactive.out
    sacrebleu ${data_path_tgt}/${i}/valid06.ch.detok -i ${results_path}/${i}.del.interactive.back.out.detok --tokenize zh -b -w 2
    
    echo 'swap'
    data_path_tgt=$code_dir/data/noisy_test/noisy_data_chen/swap
    data_path_src=$code_dir/$task/noisy/swap
    ${eval_path}/multi-bleu.perl -lc nist_valid06_test02-08/mt06_trg_u8.tok.sb  <${results_path}/${i}.swap.interactive.out
    sacrebleu ${data_path_tgt}/${i}/valid06.ch.detok -i ${results_path}/${i}.swap.interactive.back.out.detok --tokenize zh -b -w 2

    echo 'replace_src'
    data_path_tgt=$code_dir/data/noisy_test/noisy_data_chen/replace_src
    data_path_src=$code_dir/$task/noisy/replace_src
    ${eval_path}/multi-bleu.perl -lc nist_valid06_test02-08/mt06_trg_u8.tok.sb  <${results_path}/${i}.rep.interactive.out
    sacrebleu ${data_path_tgt}/${i}/valid06.ch.detok -i ${results_path}/${i}.swap.interactive.back.out.detok --tokenize zh -b -w 2

    echo 'insert'
    data_path_tgt=$code_dir/data/noisy_test/noisy_data_chen/insert
    data_path_src=$code_dir/$task/noisy/insert
    ${eval_path}/multi-bleu.perl -lc nist_valid06_test02-08/mt06_trg_u8.tok.sb  <${results_path}/${i}.ins.interactive.out
    sacrebleu ${data_path_tgt}/${i}/valid06.ch.detok -i ${results_path}/${i}.ins.interactive.back.out.detok --tokenize zh -b -w 2
}
done
