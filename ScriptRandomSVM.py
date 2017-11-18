# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:16:25 2017

@author: User
"""

from __future__ import print_function
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import accuracy_score
import nltk
from textblob import TextBlob
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import random
 

filename = 'Inception.txt'
data = open(filename).read()
 
#regex parsing into data that is readable
script = re.sub('(        )+', '\n', data)
 
leading_space_re = re.compile('^ +')
 
block_types = {
        2: 'LINE:      ',
        7: 'DIRECTION: ',
        8: 'DIRECTION: ',
        4: 'CHARACTER: ',
        }
 
##creating G Frame
g = []
 
##Identifying spaces for the sake of splitting Characters, Lines, Directions, and other unknown.
for line in script.split('\n'):
    # number of leading spaces designates block type
    match = leading_space_re.match(line)
    count = 0 if match is None else len(match.group())
    current_block = block_types.get(count, 'UNKNOWN:   ')
    g.append(tuple([current_block, count, line]))   
 
 
df = pd.DataFrame(g)
 
#limiting to only lines and characters
new_df = df[df[1].isin([2,4])]
##print(new_df)
 
cols =['TYPE','ID','OT']
new_df.columns =cols
 
##Parsing If Statements to run through character to line pick up
j,q,x,y,z = [],[],[],[],[]
 
for id1,id2,id3,id4,id5,id6,id7 in zip(new_df.iloc[23:].iterrows(),new_df.iloc[24:].iterrows(),new_df.iloc[25:].iterrows(),new_df.iloc[26:].iterrows(),new_df.iloc[27:].iterrows(),new_df.iloc[28:].iterrows(),new_df.iloc[29:].iterrows()):
    if id1[1]['ID']==4:
        if id2[1]['ID']==2:
            if id3[1]['ID']==4:
                j.append((tuple([id1[1]['OT'],id2[1]['OT'],'','','',''])))
               
    if id1[1]['ID']==4:
        if id2[1]['ID']==2:
            if id3[1]['ID']==2:
                if id4[1]['ID']==4:
                    j.append((tuple([id1[1]['OT'],id2[1]['OT'],id3[1]['OT'],'','',''])))
               
    if id1[1]['ID']==4:
        if id2[1]['ID']==2:
            if id3[1]['ID']==2:
                if id4[1]['ID']==2:
                    if id5[1]['ID']==4:
                        j.append((tuple([id1[1]['OT'],id2[1]['OT'],id3[1]['OT'],id4[1]['OT'],'',''])))
    if id1[1]['ID']==4:
        if id2[1]['ID']==2:
            if id3[1]['ID']==2:
                if id4[1]['ID']==2:
                    if id5[1]['ID']==2:
                        if id6[1]['ID']==4:
                            j.append((tuple([id1[1]['OT'],id2[1]['OT'],id3[1]['OT'],id4[1]['OT'],id5[1]['OT'],''])))
    if id1[1]['ID']==4:
        if id2[1]['ID']==2:
            if id3[1]['ID']==2:
                if id4[1]['ID']==2:
                    if id5[1]['ID']==2:
                        if id6[1]['ID']==2:
                            if id7[1]['ID']==4:
                                j.append((tuple([id1[1]['OT'],id2[1]['OT'],id3[1]['OT'],id4[1]['OT'],id5[1]['OT'],id6[1]['OT']])))                        
                            
frame = pd.DataFrame(j)
frame['COMMENT'] = frame[1]+''+frame[2] +''+frame[3]+''+frame[4]+''+frame[5]
frame = frame.rename(columns={0:'Name'})
 
##Formatting for Vectors (1)
stop = stopwords.words('english')
LM = frame[['Name','COMMENT']]
LM = pd.DataFrame(LM)
random.seed(1234)
LM = shuffle(LM)

 
##Splitting DataFrame
trainLMX = LM.iloc[:750, 0:]
validationLMX = LM.iloc[750:, 0:]

ps = PorterStemmer()
 
character_sents = []
 
##new_set_trainX = word_func(trainLMX,'COMMENT','Name')
##new_set_testX = word_func(validationLMX,'COMMENT','Name')
 
train_frameX = pd.DataFrame(trainLMX)
train_frameX = train_frameX.rename(columns = ({train_frameX.columns[0]:'Name',train_frameX.columns[1]:'COMMENT'}))
test_frameX = pd.DataFrame(validationLMX)
test_frameX = test_frameX.rename(columns = ({test_frameX.columns[0]:'Name',test_frameX.columns[1]:'COMMENT'}))


train_frameX['COMMENT'] = train_frameX.iloc[: , 1].apply(lambda x : ' '.join([word for word in x.split() if word not in (stop)])) 
train_frameX['COMMENT'] = train_frameX.iloc[: , 1].str.replace('[^\w\s]','')
train_frameX['COMMENT'] = train_frameX.iloc[: , 1].str.lower()
train_frameX['Name'] = train_frameX.iloc[: , 0].str.split('(').str[0]
train_frameX['Name'] = train_frameX.iloc[: , 0].str.strip()
test_frameX['COMMENT'] = test_frameX.iloc[: , 1].str.replace('[^\w\s]','')
test_frameX['COMMENT'] = test_frameX.iloc[: , 1].str.lower() ## Need to convert dataframe to lower case
test_frameX['COMMENT'] = test_frameX.iloc[: , 1].apply(lambda x : ' '.join([word for word in x.split() if word not in (stop)]))
test_frameX['Name'] = test_frameX.iloc[: , 0].str.split('(').str[0]
test_frameX['Name'] = test_frameX.iloc[: , 0].str.strip()

def sentence_func(df,col1,col2):
    p = []
    for i,j in zip(df[col1],df[col2]):
        sentence = sent_tokenize(i)
        for b in sentence:
            p.append(tuple([b,j]))
    return p
 
#Need to split into lowercase and removal of stopwords
    
new_set_trainX = sentence_func(train_frameX,'Name','COMMENT')
new_set_testX = sentence_func(test_frameX,'Name','COMMENT')

new_set_trainX = pd.DataFrame(new_set_trainX)
new_set_trainX = new_set_trainX.rename(columns = ({new_set_trainX.columns[0]:'Name',new_set_trainX.columns[1]:'COMMENT'}))
new_set_testX = pd.DataFrame(new_set_testX)
new_set_testX = new_set_testX.rename(columns = ({new_set_testX.columns[0]:'Name',new_set_testX.columns[1]:'COMMENT'}))

new_set_testX
names = pd.Series(['    ARIADNE',
'     ARTHUR',
'BROWNING',
'COBB',
'EAMES',
'FISCHER',
'    FLIGHT ATTENDANT',
'MAL',
'SAITO',
'YUSUF',
])
   
names_df = pd.DataFrame({'Name': names})
unique_names = pd.DataFrame({'Name':names.unique()})
unique_names['Id'] = unique_names.index

train_frameX = pd.merge(new_set_trainX, unique_names, how = 'inner', on = 'Name')
test_frameX = pd.merge(new_set_testX, unique_names, how = 'inner', on = 'Name')
test_frameX

##Build Pipeline on vectors, id vectors, and classifier for SVM
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                         alpha=1e-3, random_state = 42,
                                            max_iter=5, tol=None)),
])
                                   
text_clf.fit(train_frameX.COMMENT, train_frameX.Id)                   
predicted = text_clf.predict(test_frameX.COMMENT)
test_frameX['SVMPredict'] = predicted
np.mean(predicted==test_frameX.Id)


writer = pd.ExcelWriter('output2.xlsx')
test_frameX.to_excel(writer,'Sheet1')
writer.save()