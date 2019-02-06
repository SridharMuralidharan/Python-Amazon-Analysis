# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:10:08 2019

@author: sridh
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:53:48 2019

@author: sridh
"""


count=0

loadedjson=open('meta_Clothing_Shoes_and_Jewelry.json','r')

myproducts = {}

listofcategories = {}


for line in loadedjson:
    count += 1
    if count % 100000 == 0:
        print(count)
    myproduct = eval(line)

    myproducts[myproduct['asin']] = myproduct

    for categories in myproduct['categories']:
        for acategory in categories:
            if acategory in listofcategories:
                listofcategories[acategory] += 1
            if acategory not in listofcategories:
                listofcategories[acategory] = 1


count = 0

allnauticaasins = set()

for myproduct in myproducts:
    theproduct = myproducts[myproduct]

    count += 1
    if count % 100000 == 0:
        print(count/1503384)
    for categories in theproduct['categories']:
        for acategory in categories:
            
            if 'nautica' in acategory.lower():

                allnauticaasins.add(theproduct['asin'])

                


outputfile = open('allasins1.txt', 'w')

outputfile.write(','.join(allnauticaasins))

outputfile.close()


loadedjson=open('reviews_Clothing_Shoes_and_Jewelry.json','r')

allreviews={}
count=0

for aline in loadedjson:
    count+=1
    if count % 10000 ==0:
        print(count)
    areview=eval(aline)
    theasin=areview['asin']
    thereviewer = areview['reviewerID']
    theoverall=areview['overall']
    theid=areview['reviewerID']
    
    if theasin in allnauticaasins:
        thekey ='%s.%s.%s.%s' %(theasin,thereviewer,theoverall,theid)
        allreviews[thekey]= areview
    

len(allreviews)

import json

json.dump(allreviews,open('allnauticareviews1.json','w'))

#outputfile = open('allnauticareviews1.txt', 'w')

#outputfile.write(','.join(allreviews))

#outputfile.close()

allreviews=json.load(open('allnauticareviews.json','r'))

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

stop_words=stopwords.words('english')

#stop_words +=['nautica']

stop_words.append('nautica')

texts=set()
def load_texts(topicdata):
    for areview in topicdata:
        if 'reviewText' in topicdata[areview]:
            reviewtext= topicdata[areview]['reviewText']
            summary=topicdata[areview]['summary']
            asin=topicdata[areview]['asin']
            overall=topicdata[areview]['overall']
            reviewerid=topicdata[areview]['reviewerID']
            review='%s %s %s' %(asin,summary,reviewtext)
            rating='%s'%(overall)
            id_1='%s'%(reviewerid)
            texts.add(review)
            texts.add(rating)
            texts.add(id_1)
    
print('loading texts')
load_texts(allreviews)

documents=list(texts)

vectorizer=TfidfVectorizer(stop_words=stop_words)
X= vectorizer.fit_transform(documents)

true_k =10

model =KMeans(n_clusters=true_k,max_iter=100000)

model.fit(X)

print("Top terms per cluster")

order_centroids =model.cluster_centers_.argsort()[:,::-1]
terms= vectorizer.get_feature_names()

for i in range(true_k):
    topic_terms = [terms[ind] for ind in order_centroids[i,:4]]
    print('%d: %s' %(i,' '.join(topic_terms)))


import os
import re
outfiles={}
s=['']*13009
l=['']*13009

try:
    os.mkdir('output')
    
except OSError:
    print('directory already exists')
    
else:
    print("successfully created the dictionary")
    
for atopic in range(true_k):
    topicterms=[terms[ind]for ind in order_centroids[atopic,:4]]
    outfiles[atopic]=open(os.path.join('output','_'.join(topicterms)+'.txt'),'w')
    
outputfile = open('allnauticareviews3.txt', 'w')

for areview in allreviews:
    if 'reviewText' in allreviews[areview]:
        thereview=allreviews[areview]
        reviewwithmetadata="%s %s %s" % (thereview['asin'],thereview['summary'],thereview['reviewText'])
        overallrating="%s" %(thereview['overall'])
        reviewerid1='%s'%(thereview['reviewerID'])
        Y=vectorizer.transform([reviewwithmetadata])
        predictions = model.predict(Y)
        s.append(reviewwithmetadata)
        s.append(overallrating)
        s.append(reviewerid1)
        #l=re.sub("B0",'/t',reviewwithmetadata)
       # outfiles[prediction].write('%s\n' % review)
   
        outputfile.write("%s\n"%thereview['reviewText'])
        outputfile.write("%s\n" %thereview['overall'])
        outputfile.write("%s\n" %thereview['reviewerID'])

        #for i in range (0,13009):
            #s.append(reviewwithmetadata)

       # predictions=model.predict(Y)
        for prediction in model.predict(Y):
            outfiles[prediction].write('%s\n' % reviewwithmetadata)
            outfiles[prediction].write('%s\n' % overallrating)
            outfiles[prediction].write('%s\n'%reviewerid1)
  
       
for n,f in outfiles.items():
    f.close()
    
outputfile.close()

from sklearn.decomposition import LatentDirichletAllocation
from multiprocessing import cpu_count

k=25

lda_tfidf=LatentDirichletAllocation(n_topics=k,n_jobs=cpu_count())

lda_tfidf.fit(X)

#import pyLDAvis
##pip install pyldavis
#import pyLDAvis.sklearn
#pyLDAvis.enable_notebook()
#
#p=pyLDAvis.sklearn.prepare(lda_tfidf,X,vectorizer)
#pyLDAvis.save_html(p,'pyLdavis.html') 
#


import numpy as np   
import pandas as pd  
  
# Import dataset 


dataset_new1 = pd.read_csv('allnauticareviews3.txt', delimiter = '\t')  

#from nltk.stem.porter import PorterStemmer 
#pip install vaderSentiment

# Creating the Bag of Words model 



#using Vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))



for i in range(0,len(dataset_new1)):
    if (i%2!=0 or i%3==0):
        sentiment_analyzer_scores(dataset_new1.iloc[i][0])
    
    
    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyze = SentimentIntensityAnalyzer()

s1 = []
s2 = []
s3 = []
s4 = []
s5 = []
s6 = []
mean_rating=[]
#sentiment analysis for testing the polarity and subjectivity
for i in range(0,len(dataset_new1)):
    if ((i+1)%3 ==0): 
        s1.append(TextBlob(dataset_new1.iloc[i][0]).polarity)
        s2.append(TextBlob(dataset_new1.iloc[i][0]).subjectivity)


#sentiment analysis
for i in range(0,len(dataset_new1)):
    if ((i+1)%3 ==0):
        s3.append(analyze.polarity_scores(dataset_new1.iloc[i][0])['neg'])
        s4.append(analyze.polarity_scores(dataset_new1.iloc[i][0])['pos'])
        s5.append(analyze.polarity_scores(dataset_new1.iloc[i][0])['neu'])
        s6.append(analyze.polarity_scores(dataset_new1.iloc[i][0])['compound'])    

#rating for each product  
#import statistics
for i in range(0,len(dataset_new1)):
    if i%3 ==0:
        mean_rating.append(dataset_new1.iloc[i][0])
        
       
reviewer_id=[]
#list of reviwerids
for i in range(0,len(dataset_new1)):
    if ((i-1)%3 ==0):
        reviewer_id.append(dataset_new1.iloc[i][0])
        
 #to find the super reviwer or the reviwer with maximum number of reviews
str1=[str(i) for i in reviewer_id]
   
from collections import Counter
Counter(str1)
   
dict((i, reviewer_id.count(i)) for i in reviewer_id)        

#the reviewer with most reviews 'A3IGJJVPKOYO1N': 3,


TextBlob(" nice slippers fit slipper").sentiment

TextBlob(" small size run return").sentiment
TextBlob(" well made slippers comfortable").sentiment
TextBlob(" slippers great quality product").sentiment
TextBlob("  pair years slippers second").sentiment
TextBlob(" recommend would highly great").sentiment
TextBlob("  house around comfortable shoes").sentiment
TextBlob(" best slippers ever owned").sentiment
TextBlob("  slipper great comfortable size").sentiment
TextBlob("  wide width eee slippers").sentiment
TextBlob("  size 12 13 shoe").sentiment




