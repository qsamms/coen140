import numpy as np
import pandas as pd
import re
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
import time

# This function creates a file in the same format as train.dat given already 
# pre-processed data
def makeProcessedFile(fileName,processed_docs,nums):
    stemFile = open(fileName,"a")

    for i in range(len(processed_docs)):
        docString = ' '.join(processed_docs[i])
        stemFile.write(nums[i] + ' ' + docString + '\n')

# This function reads already preprocessed data from a given fille
def readProcessed(fileName):
    processed_docs = []
    nums = []
    stemFile = open(fileName,'r')

    for line in stemFile:
        num = line[0]
        text = line[2:]
        lst = text.split()
        processed_docs.append(lst)
        nums.append(num)

    return processed_docs, nums

# this function reads the lines from a file and preforms 
#stemming on each line, in addition it removes special 
# characters and filters out stop words, and removes 1 character words
def readAndProcessStem(fileName,train=True):
    processed_docs = []
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    file = open(fileName,"r")

    if train:
        nums = [] 
        for line in file:
            classNum = line[0]
            nums.append(classNum)
            text = line[2:]
            nospecialchar = re.sub(r'[^a-zA-Z0-9_]', ' ', text)
            words = word_tokenize(nospecialchar)
            stemmed_words = [stemmer.stem(word) for word in words]
            filtered_words = [word for word in stemmed_words if word not in stop_words]
            for word in filtered_words:
                if len(word) == 1:
                    filtered_words.remove(word)
            processed_docs.append(filtered_words)

        return processed_docs, nums
    
    else:
        for line in file:
            nospecialchar = re.sub(r'[^a-zA-Z0-9_]', ' ', line)
            words = word_tokenize(nospecialchar)
            stemmed_words = [stemmer.stem(word) for word in words]
            filtered_words = [word for word in stemmed_words if word not in stop_words]
            for word in filtered_words:
                if len(word) == 1:
                    filtered_words.remove(word)
            processed_docs.append(filtered_words)
        
        return processed_docs

# This function reads the lines from a file and preforms lemmatization
# on each line, in addition removes special characters and filters out stop words
# removes 1 character words aswell
def readAndProcessLemma(fileName,train=True):
    processed_docs = []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    file = open(fileName,'r')

    if train:
        nums = []
        for line in file:
            classNum = line[0]
            nums.append(classNum)
            text = line[2:]
            nospecialchar = re.sub(r'[^a-zA-Z0-9_]', ' ', text)
            words = word_tokenize(nospecialchar)
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            filtered_words = [word for word in lemmatized_words if word not in stop_words]
            for word in filtered_words:
                if len(word) == 1:
                    filtered_words.remove(word)
            processed_docs.append(filtered_words)

        return processed_docs, nums
    
    else:
        for line in file:
            nospecialchar = re.sub(r'[^a-zA-Z0-9_]', ' ', line)
            words = word_tokenize(nospecialchar)
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            filtered_words = [word for word in lemmatized_words if word not in stop_words]
            for word in filtered_words:
                if len(word) == 1:
                    filtered_words.remove(word)
            processed_docs.append(filtered_words)

        return processed_docs

# This function builds a sparce matrix out of a list of documents
def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    #number of rows will be the length of the 2D array
    nrows = len(docs)
    #dictionary to hold key,val pairs for a word and its associated encoding
    idx = {}
    #id number associated with each distinct word in the document set
    tid = 0
    #number of distinct words
    nnz = 0
    #this loop gives each word a unique id 
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    #number of columns is the total number of distinct words in our dictionary 
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        #counts how many occurances of each word in a single document and stores it in 
        #key, val format
        cnt = Counter(d)
        #cnt.most_common() method returns each key,val pair in order of most to least common
        #keys is a list of the words in this document d in order of most to least common
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        #enumerate assinges an index starting at 0 for each key
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    #builds a csr_matrix
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    return mat

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat

# This function calculates the class categorization of a given document in the test 
# set using cosine similarities. Uses majority vote to categorize and tie breaks
# using highest aggregate similarity
def classifyCosine(bothmat,trmat,trcls,id,k):
    # gets the row of the specified (doc) argument from its id(index in docs list)
    x = bothmat[id,:]
    dots = x.dot(trmat.T)
    sims = list(zip(dots.indices, dots.data))
    sims.sort(key=lambda x: x[1], reverse=True)
    tc = Counter(trcls[s[0]] for s in sims[:k]).most_common(2)
    #majority vote
    if len(tc) < 2 or tc[0][1] > tc[1][1]:
        return tc[0][0]
    
    #tie breaker
    tc = defaultdict(float)
    for s in sims[:k]:
        tc[trcls[s[0]]] += s[1]
    
    return sorted(tc.items(), key = lambda x : x[1],reverse = True)[0][0]
    
def main():
    outFile = open('outputStem5.dat','a')
    #docs,classes = readAndProcess("train.dat")
    #makeProcessedFile("stemmed.dat",docs,classes)
    #traindocs,trcls = readProcessed('stemmed.dat')
    traindocs,trcls = readAndProcessStem("train.dat")
    testdocs = readAndProcessStem('test.dat',train=False)

    bothdocs = traindocs.copy()
    for doc in testdocs:
        bothdocs.append(doc)

    bothmat = build_matrix(bothdocs)
    csr_l2normalize(bothmat)
    trmat = bothmat[:len(traindocs)]
    #classify the test data from 5 nearest neighbors
    idx = 102080
    for doc in testdocs:
        num = classifyCosine(bothmat,trmat,trcls,idx,5)
        idx += 1
        outFile.write(str(num) + '\n')


start = time.time()
main()
end = time.time()
print(f"{end - start} seconds")