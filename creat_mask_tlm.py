import sys
import random
file_in = sys.argv[1]
file_tgt =sys.argv[2]
file_align = sys.argv[3]
file_out= sys.argv[4]
file_label = sys.argv[5]

MASK_TOKEN = '[MASK]'
SEP_TOKEN = '[SEP]'
MASK_RATIO = sys.argv[6]
f1 =  open(file_out, 'w', encoding='utf-8')

f2 =  open(file_label, 'w', encoding='utf-8')
with open( file_in, 'r',encoding='utf-8') as f3, open( file_tgt, 'r',encoding='utf-8') as f4, open( file_align, 'r',encoding='utf-8') as f5: 
        for line, line_tgt, align in zip(f3,f4,f5):
            line = line.strip()
            line_tgt = line_tgt.strip().split()
            #print(line)
            #print(align)
            if align != '\n':
                align = align.strip().split('|')[:-1]
                total_mask_num = int(len(align) * float(MASK_RATIO) +0.5)
                numbers = [int(i) for i in range(len(align))]
                choiced_idx = random.sample(numbers, total_mask_num)
                #print(align)
                for i in choiced_idx:
                    a = align[i]
                    a = a.strip().split('-')
                    #print(a)
                    start_idx = int(a[2])
                    end_idx = int(a[3]) 
                    #print(start_idx,end_idx)
                    label = line_tgt[start_idx: end_idx]
                    label = ' '.join(label)
                    #print(label)
                    new_line = line_tgt[:start_idx]
                    new_line.append(MASK_TOKEN)
                    new_line += line_tgt[end_idx:]
                    new_line = ' '.join(new_line)
                    new_line = line +' ' + SEP_TOKEN + ' ' + new_line
                    f2.write(label+ '\n')
                    f1.write(new_line+ '\n')

        
