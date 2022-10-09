import argparse
import torch
from fairseq.models.transformer import TransformerModel
import sacrebleu
import numpy as np
import random
import os

def parseargs():
    parser = argparse.ArgumentParser(description="Adversial Example")
    parser.add_argument("--s2t_model", type=str, required=True)
    parser.add_argument("--t2s_model", type=str, required=True)
    parser.add_argument("--mlm_model", type=str, required=True)
    parser.add_argument("--tlm_model", type=str, required=True)

    parser.add_argument("--s2t_codes", type=str, required=True)
    parser.add_argument("--t2s_codes", type=str, required=True)
    parser.add_argument("--mlm_codes", type=str, required=True)
    parser.add_argument("--tlm_codes", type=str, required=True)

    parser.add_argument("--s2t_data", type=str, required=True)
    parser.add_argument("--t2s_data", type=str, required=True)
    parser.add_argument("--mlm_data", type=str, required=True)
    parser.add_argument("--tlm_data", type=str, required=True)

    parser.add_argument("--src_file", type=str, required=True)
    parser.add_argument("--tgt_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--align_file", type=str, required=True)
    parser.add_argument("--stopword_file", type=str, required=True)

    parser.add_argument("--replace_rate", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.)
    parser.add_argument("--delta", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--continue_num", type=int, default=0)
    return parser.parse_args()


def bleu_score(src_out,ref):
    bleu = sacrebleu.corpus_bleu(src_out, ref)
    #bleu = sacrebleu.sentence_bleu(src_out, [ref],smooth_method='exp')
    return bleu.score

def mlm(src_words, mlm_model,k):
    split_text = src_words.split()
    x_len = len(split_text)
    mask_texts = []
    for l in range(x_len):
        mask_text = split_text[ : l] + ['[MASK]'] + split_text[min(l + 1, x_len):]
        mask_text = ' '.join(mask_text)
        mask_texts.append(mask_text)
    sim_words = mlm_model.translate(mask_texts, beam= k+1 ,n_best=k+1)
    for i, words in enumerate(sim_words):
        flag = 0
        for j, word in enumerate(words):
            if word == split_text[i]:
                flag =1
                break
        if flag == 0:
            sim_words[i] = words[: k]
        else:
            sim_words[i].remove(split_text[i])
    return sim_words

def tlm(src_words, tlm_model):
    return tlm_model.translate_one(src_words)

def attack(id, x_ref, y_ref, src2tgt, tgt2src, replace_rate, mlm_model, tlm_model, align, k, gamma, delta, alpha,stop_words,out_file):
    src_words = x_ref.split()
    src_words_new = src_words.copy()
    tgt_words = y_ref.split()
    tgt_words_new = tgt_words.copy()
    x_len = len(src_words)
    total_replace_words = int(x_len * replace_rate + 0.5)
    if total_replace_words == 0:
        return
    # round-trip for x_ref
   
    y = src2tgt.translate_one(x_ref)
    x_ref_hat = tgt2src.translate_one(y)
    sim_x_ref = bleu_score(x_ref_hat, x_ref)  # sim(x_ref, \hat{x_ref})

    # round-trip for y_ref
    x = tgt2src.translate_one(y_ref)
    y_ref_hat = src2tgt.translate_one(x)
    sim_y_ref = bleu_score(y_ref_hat, y_ref)  # sim(y_ref, \hat{y_ref})

    if sim_x_ref < delta or sim_y_ref < delta:
        return

    # find synonyms

    sim_words = mlm(x_ref, mlm_model,k)
    # print(sim_words)
    if not sim_words:
        return

    synonyms_all = sim_words
    # start replacing and attacking
    n = 0  # nums of words has be replaced

    # alignment for original src and tgt
    alignment = align.strip().split('|')[:-1]
    src_align = []
    tgt_align = []
    for a in alignment:
        a = a.split('-')
        src_align.append((int(a[0]), int(a[1])))
        tgt_align.append((int(a[2]), int(a[3])))
    mask = []
    for idx, word in enumerate(src_words):
        if  word not in stop_words and (idx, idx + 1) in src_align:
            mask.append(True)
        else:
            mask.append(False)

    x_adv = x_ref
    y_ref_new = y_ref
    while n < total_replace_words and True in mask:
        out_file.write("S-{}-{}\t{}".format(id,n,x_adv)+'\n')
        out_file.write("T-{}-{}\t{}".format(id,n,y_ref_new)+'\n')
        tmp_src = src_words.copy()
        mask_id = []
        for m_idx, m in enumerate(mask):
            if m == True:
                mask_id.append(m_idx)
        replace_bleu = np.ones((x_len, k)) * np.inf  # initialize each position's bleu to inf
        x_adv_all = []
        for i, syn in enumerate(synonyms_all):
            if mask[i] != False:
                if syn == ' ':
                    mask[i] = False
                else:
                    new_src = src_words_new.copy()
                    for s in syn:
                        new_src[i] = s
                        x_adv = ' '.join(new_src)
                        x_adv_all.append(x_adv)
        y_adv_all = src2tgt.translate_one(x_adv_all)
        x_adv_all_hat = tgt2src.translate_one(y_adv_all)
        z = 0
        for x_adv, x_adv_hat in zip(x_adv_all, x_adv_all_hat):
            word_bleu =bleu_score(x_adv, x_adv_hat)
            replace_bleu[mask_id[z//k], z%k] = word_bleu
            z += 1
        replace_position = np.where(replace_bleu == replace_bleu.min())
        r = random.randint(0,len(replace_position[0])-1)
        replace_position_x_i, replace_position_x_j = int(replace_position[0][r]), int(replace_position[1][r])
        idx = src_align.index((replace_position_x_i, replace_position_x_i + 1))
        # replace y_ref
        replace_position_y_i, replace_position_y_j = tgt_align[idx]
        tmp_tgt = tgt_words[:replace_position_y_i]
        tmp_tgt.append('[MASK]')
        tmp_tgt += tgt_words[replace_position_y_j:]
        src_words_new[replace_position_x_i] = synonyms_all[replace_position_x_i][replace_position_x_j]
        tmp_src[replace_position_x_i] = synonyms_all[replace_position_x_i][replace_position_x_j]
        mask[replace_position_x_i] = False
        y_ref_tmp = ' '.join(tmp_tgt)
        x_ref_tmp = ' '.join(tmp_src)
        join_x_y = x_ref_tmp + ' [SEP] ' + y_ref_tmp
        replace_token_y = tlm(join_x_y, tlm_model)

        t = tgt_words_new[:replace_position_y_i]
        t.append(replace_token_y)
        t += tgt_words_new[replace_position_y_j:]
        tgt_words_new = t
        mask[replace_position_x_i] = False
        
        x_adv = ' '.join(src_words_new)
        y_ref_new = ' '.join(tgt_words_new)
        n += 1
   
    x_adv = ' '.join(src_words_new)
    y_ref_new = ' '.join(tgt_words_new)
    
    y_adv = src2tgt.translate_one(x_adv) 
    x_adv_hat = tgt2src.translate_one(y_adv)
    y_adv_hat = src2tgt.translate_one(x_adv_hat)

    sim_x_adv = bleu_score(x_adv, x_adv_hat)
    sim_y_adv = bleu_score(y_adv, y_adv_hat)    
    
    E1 = (sim_x_ref - sim_x_adv) / (sim_x_ref )
    E2 = (sim_y_ref - sim_y_adv) / (sim_y_ref )
    out_file.write("D-{}\t{}\t{}".format(id,E1,x_adv)+'\n')
    out_file.write("H-{}\t{}\t{}".format(id,E2,y_ref_new)+'\n')
    out_file.flush()
    return x_adv,y_ref_new

def main(param):
    s2t = TransformerModel.from_pretrained(
          param.s2t_data,
          checkpoint_file=param.s2t_model,
          data_name_or_path='.',
          bpe='subword_nmt',
          bpe_codes= param.s2t_codes
          )

    t2s = TransformerModel.from_pretrained(
          param.t2s_data,
          checkpoint_file=param.t2s_model,
          data_name_or_path='.',
          bpe='subword_nmt',
          bpe_codes= param.t2s_codes
          )

    mlm = TransformerModel.from_pretrained(
          param.mlm_data,
          checkpoint_file=param.mlm_model,
          data_name_or_path='.',
          bpe='subword_nmt',
          bpe_codes= param.mlm_codes
          )

    tlm = TransformerModel.from_pretrained(
           param.tlm_data,
          checkpoint_file=param.tlm_model,
          data_name_or_path='.',
          bpe='subword_nmt',
          bpe_codes= param.tlm_codes
          )

    s2t.cuda()
    t2s.cuda()
    mlm.cuda()
    tlm.cuda()
    
    alignments = open(params.align_file, 'r', encoding='utf-8')
    src_file = open(param.src_file, 'r', encoding='utf-8')
    tgt_file = open(param.tgt_file, 'r', encoding='utf-8')
    out_file = open(param.out_file, 'w', encoding='utf-8')
    print(param.out_file)
    stop_words = np.load(params.stopword_file)
    i = 0
    adv_examples = []
    #print(params.continue_num)
    for src, tgt, align in zip(src_file, tgt_file, alignments):
        i += 1
        
        if i > int(params.continue_num):
            #print(i)
            #print(param.out_file)
            x_ref = src.strip()
            y_ref = tgt.strip()
            attack(i, x_ref, y_ref, s2t, t2s, param.replace_rate, mlm, tlm, align, param.k, param.gamma, param.delta, param.alpha, stop_words,out_file)
            #print(adv_x)
    out_file.close()


if __name__=="__main__":
    #print(torch.cuda.current_device())    
    params = parseargs()
    main(params)
