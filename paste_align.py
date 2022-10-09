import sys
src_file = sys.argv[1]
tgt_file = sys.argv[2]
out_file = sys.argv[3]
f3 = open(out_file, 'w', encoding='utf-8') 
with open(src_file, 'r', encoding='utf-8') as f1, open(tgt_file, 'r', encoding='utf-8') as f2:
        for line1, line2 in zip(f1,f2):
            line = line1.strip() + ' ||| ' +line2.strip()
            f3.write(line + '\n')

