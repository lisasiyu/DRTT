import sys

file_src = sys.argv[1]
file_tgt = sys.argv[2]

score_x_1 = sys.argv[3]
score_x_2 = sys.argv[4]
score_y = sys.argv[5]

f1 =  open(file_src,'r',encoding='utf-8') 
f2 = open(file_tgt,'r',encoding='utf-8')
for line1, line2 in zip (f1,f2):
    line1 = line1.strip().split()
    line2 = line2.strip().split()
    x = line1[0]
    y = line2[0]

    if float(x)>= float(score_x_1) and float(x)<= float(score_x_2) and float(y)<= float(score_y):
        print("D-\t{}".format(' '.join(line1[1:])))
        print("H-\t{}".format(' '.join(line2[1:])))
    
