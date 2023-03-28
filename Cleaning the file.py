
import pickle
import re
import glob
from pathlib import Path
import csv
import pandas as pd
import os


def clean_corpus(sentence):
    sentence = re.sub(r'^-','',sentence) #remove dialogue hyphens at the beginning of a line
    sentence = re.sub(r'\n','',sentence) #remove line breaks
    sentence = sentence.replace('\u2014', '') #remove em-Dash dialogue hyphens at the beginning of a line
    sentence = sentence.replace('\u2012', '') 
    sentence = sentence.replace('\u2013', '')
    sentence = sentence.replace('\u2015', '')  
    return sentence

number = 0

for file in glob.iglob('/media/amaia/TOSHIBA EXT/Corpus_dividido/*'):
    print(file)
    corpus_clean = []
    with open(file,'r') as txt:
        for line in txt.readlines():
            sentence = line.lstrip()
            if not sentence.startswith(('#')):
                if not sentence.startswith(('[')):
                    if not sentence.startswith(('(')):
                        if not sentence.startswith(('\u266a')):
                            sentence = clean_corpus(sentence)
                            corpus_clean.append(sentence.strip()) 
    dict = {'text': corpus_clean}
    dataframe = pd.DataFrame(dict)   
    dataframe.to_csv(f'/media/amaia/TOSHIBA EXT/Clean/{number}.csv') 
    number += 1
