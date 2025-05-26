import sys
import pandas as pd
import difflib
from evalchem import evalverb
from nltk.translate.bleu_score import sentence_bleu

# input result path
path = ''
results = pd.read_csv(path)
inps = results['reaction_name'].tolist()
anss = results['action_labels'].tolist()
preds = results['action_preds_1'].tolist()

sm1 = 0
sm2 = 0
bleu2 = 0
bleu4 = 0
dup = {}
ski = {}
em = 0

for i in range(len(anss)):
    acc, verbacc, r1, r2, pa1, pa2 = evalverb(' '+preds[i*1].strip('\n'), ' '+anss[i].strip('\n'))
    for wd in r1:
        if wd not in dup:
            dup[wd] = 0
        dup[wd]+=1
    for wd in r2:
        if wd not in ski:
            ski[wd] = 0
        ski[wd]+=1
    
    if preds[i] == anss[i]:
        em+=1

    sm1+=acc
    sm2+=verbacc
    bleu2 += sentence_bleu([anss[i].split(' ')], preds[i*1].split(' '), weights=[0,1,0,0])
    bleu4 += sentence_bleu([anss[i].split(' ')], preds[i*1].split(' '), weights=[0,0,0,1])

    seq=difflib.SequenceMatcher(None, preds[i],anss[i])
    if seq.ratio()*100 > 80:
        em += 1
    print(seq.ratio()*100)
    
print(dup, ski)
print(f'SM1: {100*sm1/len(anss):.2f}')
print(f'SM2: {100*sm2/len(anss):.2f}')
print(f'BLEU-2: {100*bleu2/len(anss):.2f}')
print(f'BLEU-4: {100*bleu4/len(anss):.2f}')
print(f'EM: {100*em/len(anss):.2f}')


