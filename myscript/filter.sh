code_dir=
data_dir=
task=
file_in=
file_out=
export PYTHONPATH=$code_dir:$PYTHONPATH
src=ch
tgt=en

if [ ! -d $file_out ];then
    mkdir -p $file_out
    chmod -R 777 $file_out
fi

grep "^D-" $file_in/out | cut -f2- >  $file_out/tmp.src
grep "^H-" $file_in/out | cut -f2- >  $file_out/tmp.tgt
PYTHONIOENCODING=utf-8 python $code_dir/filter.py  $file_out/tmp.src  $file_out/tmp.tgt 0.01 1. 0.5 >  $file_out/tmp.filter
grep "^D-"  $file_out/tmp.filter | cut -f2- > $file_out/file_out.adv.$src
grep "^H-"  $file_out/tmp.filter | cut -f2- > $file_out/file_out.adv.$tgt

cat $file_out/file_out.adv.$src $data_dir/train_bpe30k.ch.shuf.delbpe > $file_out/train.all.$src
cat $file_out/file_out.adv.$tgt $data_dir/train_bpe30k.en.shuf.delbpe > $file_out/train.all.$tgt
BPEROOT=$code_dir/subword-nmt/subword_nmt
BPE_TOKENS=30000
SCRIPTS=$code_dir/mosesdecoder/scripts
CLEAN=$SCRIPTS/file_outing/clean-corpus-n.perl
BPE_CODE_CH=$file_out/codes_ch
BPE_CODE_EN=$file_out/codes_en

echo "learn_bpe.py"
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $file_out/file_out.all.en > $BPE_CODE_EN
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $file_out/file_out.all.ch > $BPE_CODE_CH

echo "apply_bpe.py"

python $BPEROOT/apply_bpe.py -c $BPE_CODE_CH < $data_dir/file_out_bpe30k.ch.shuf.delbpe > $file_out/train.org.bpe.ch
python $BPEROOT/apply_bpe.py -c $BPE_CODE_EN < $data_dir/file_out_bpe30k.en.shuf.delbpe > $file_out/train.org.bpe.en

python $BPEROOT/apply_bpe.py -c $BPE_CODE_CH < $file_out/file_out.adv.ch > $file_out/train.adv.bpe.ch
python $BPEROOT/apply_bpe.py -c $BPE_CODE_EN < $file_out/file_out.adv.en > $file_out/train.adv.bpe.en

cat $file_out/file_out.adv.bpe.ch $file_out/train.org.bpe.ch > $file_out/train.all.bpe.ch
cat $file_out/file_out.adv.bpe.en $file_out/train.org.bpe.en > $file_out/train.all.bpe.en

sample_array=(2 3 4 5 6 8)
file_test=$file_out/test
if [ ! -d $file_test ];then
    mkdir -p $file_test
    chmod -R 777 $file_test
fi

for i in ${sample_array[*]};
do
    delbpe_file=$data_dir/nist_valid06_test02-08/mt0${i}_src_u8.BPE.delbpe
    python $BPEROOT/apply_bpe.py -c ${BPE_CODE_CH} < ${delbpe_file} > $file_test/mt0${i}_src_u8.BPE.own

done
python $code_dir/shuffle_corpus.py --corpus $file_out/file_out.all.bpe.ch $file_out/train.all.bpe.en

mv $file_out/file_out.all.bpe.ch.shuf $file_out/train.ch
mv $file_out/file_out.all.bpe.en.shuf $file_out/train.en

for i in {0,0.1,0.2,0.3,0.4,0.5}
do

#: <<'comment'
    out_file=$file_out/noisy/replace_both/$i
    #echo "apply_bpe.py"
    if [ ! -d $out_file ];then
        mkdir -p $out_file
        chmod -R 777 $out_file
    fi
    noisy_text=$code_dir/data/noisy_test/noisy_data_chen/replace_both/$i

    python $BPEROOT/apply_bpe.py -c $BPE_CODE_CH < $noisy_text/valid06.ch > $out_file/valid06.bpe.ch


    out_file=$file_out/noisy/delete/$i
    #echo "apply_bpe.py"
    if [ ! -d $out_file ];then
        mkdir -p $out_file
        chmod -R 777 $out_file
    fi
    noisy_text=$code_dir/data/noisy_test/noisy_data_chen/delete/$i

    python $BPEROOT/apply_bpe.py -c $BPE_CODE_CH < $noisy_text/valid06.ch > $out_file/valid06.bpe.ch

    out_file=$file_out/noisy/swap/$i
    #echo "apply_bpe.py"
    if [ ! -d $out_file ];then
        mkdir -p $out_file
        chmod -R 777 $out_file
    fi
    noisy_text=$code_dir/data/noisy_test/noisy_data_chen/swap/$i

    python $BPEROOT/apply_bpe.py -c $BPE_CODE_CH < $noisy_text/valid06.ch > $out_file/valid06.bpe.ch
    out_file=$file_out/noisy/replace_src/$i
    #echo "apply_bpe.py"
    if [ ! -d $out_file ];then
        mkdir -p $out_file
        chmod -R 777 $out_file
    fi
    noisy_text=$code_dir/data/noisy_test/noisy_data_chen/replace_src/$i

    python $BPEROOT/apply_bpe.py -c $BPE_CODE_CH < $noisy_text/valid06.ch > $out_file/valid06.bpe.ch
    out_file=$file_out/noisy/insert/$i
    #echo "apply_bpe.py"
    if [ ! -d $out_file ];then
        mkdir -p $out_file
        chmod -R 777 $out_file
    fi
    noisy_text=$code_dir/data/noisy_test/noisy_data_chen/insert/$i

    python $BPEROOT/apply_bpe.py -c $BPE_CODE_CH < $noisy_text/valid06.ch > $out_file/valid06.bpe.ch
done
