import _pickle as pickle
import os
import sys
# sys.path.append('../coco_caption')
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('code'))))
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.meteor.meteor import Meteor

'''
    To evaluate with pycocoevalcap
    input data should be formed like
    {id: [sentence], ...}
    
    ex) 
        predict: 
            {0: ['This is a sample'], 1: ['This is a cat'], ...}
        refer:
            {0: ['This is a sample',
                 'This is samples',
                 '...',
                 '...',
                 '...'], 
             1: ['This is a cat', ...], 
             ...}
'''

def score(ref, hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"])
        ,(Meteor(),"METEOR")
#         ,(Rouge(),"ROUGE_L"),
#         (Cider(),"CIDEr")
    ]
    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores
    

def evaluate(data_path='./data', split='val', date='20190930', get_scores=False):
    reference_path = os.path.join(data_path, "%s/%s/%s.references.pkl" %(split, date, split))
    predicted_path = os.path.join(data_path, "%s/%s/%s.predicted.captions.pkl" %(split, date, split))
    
    # load caption data
    with open(reference_path, 'rb') as f:
        ref = pickle.load(f)
    with open(predicted_path, 'rb') as f:
        pred = pickle.load(f)
    
    # make dictionary
    hypo = {}
    for i, caption in enumerate(pred):
        hypo[i] = [caption]
    
#     ref = {}
#     for i, caption in enumerate(real):
#         ref[i] = [caption]
  
    # compute bleu score
    final_scores = score(ref, hypo)
    
    # print out scores
    print('Bleu_1:\t',final_scores['Bleu_1'])  
    print('Bleu_2:\t',final_scores['Bleu_2'])  
    print('Bleu_3:\t',final_scores['Bleu_3'])  
    print('Bleu_4:\t',final_scores['Bleu_4'])  
    print('METEOR:\t',final_scores['METEOR'])  
#     print 'ROUGE_L:',final_scores['ROUGE_L']  
#     print 'CIDEr:\t',final_scores['CIDEr']
    
    if get_scores:
        return final_scores
    