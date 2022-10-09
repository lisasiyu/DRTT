import sys
import random

file_in = sys.argv[1]
file_out = sys.argv[2]
file_label = sys.argv[3]
MASK_TOKEN = '[MASK]'
MASK_RATIO = sys.argv[4]
f1 =  open(file_out, 'w', encoding='utf-8')
f2 =  open(file_label, 'w', encoding='utf-8')
with open( file_in, 'r',encoding='utf-8') as f3: 
        for line in f3:
            line = line.strip().split()
            len_line = len(line)
            total_mask_num = int(len_line * float(MASK_RATIO) + 0.5)
            numbers = [int(i) for i in range(len_line)]
            choiced_idx = random.sample(numbers, total_mask_num)
            for i in choiced_idx:
                label = line[i]
                new_line = line[:i]
                new_line.append(MASK_TOKEN)
                if i < len_line-1:
                    new_line += line[i+1:]
                new_line = ' '.join(new_line)
                f1.write(new_line + '\n')
                f2.write(label + '\n')

        
