import os
import io
import math
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()
words_in_stop = set(stopwords.words('english'))
document = {}
words_in_doc = {}


def main():

    #Fetching the documents from the folder and storing in dictionary
    corpusroot = './presidential_debates'
    for txt_filename in os.listdir(corpusroot):
        file = io.open(os.path.join(corpusroot, txt_filename), "r", encoding='UTF-8')
        document[txt_filename] = file.read()
        file.close()
        document[txt_filename] = document[txt_filename].lower()

    #Fetching the tokens and storing it in dictionary.
    for key,value in document.items():
        new_array = []
        tokens = tokenizer.tokenize(document[key])
        for t in tokens:
            if t not in words_in_stop:
                new_array.append(stemmer.stem(t))
        new_array.sort()
        words_in_doc[key] = dict.fromkeys(new_array, 0)

    #Here we are counting the no of occurences of each token in the Document
    for key,value in document.items():
        tokens = tokenizer.tokenize(document[key])
        for element,cost in words_in_doc[key].items():
            token_count = 0
            for t in tokens:
                if element in t:
                    token_count = token_count + 1
            words_in_doc[key][element] = token_count

    (name_of_document, rank) = query("health insurance wall street")
    print ("name_of_document, rank", name_of_document, rank)


# Function to compute idf of a token in the document
def getidf(name_of_token):
    idf = -1
    ct_of_token_in_doc = 0
    no_of_docs = 0
    for key,value in document.items():
        no_of_docs = no_of_docs + 1
        for element,cost in words_in_doc[key].items():
            if name_of_token == element:
                ct_of_token_in_doc = ct_of_token_in_doc + 1
    if (ct_of_token_in_doc > 0):
        idf = math.log10(no_of_docs / float(ct_of_token_in_doc))
    return idf


#Function to compute tf of a token in the document
def gettf(name_of_doc, name_of_token):
    tf = 0
    count = 0


    for element,cost in words_in_doc[name_of_doc].items():
        if element == name_of_token:
            count = words_in_doc[name_of_doc][name_of_token]
            break

    if (count > 0):
        tf = 1 + math.log10(count)
    return tf

def norm_weight(name_of_doc, name_of_token):
    element_square = 0

    for element,cost in words_in_doc[name_of_doc].items():
        tottf = gettf(name_of_doc, element)
        totidf = getidf(element)
        element_weight = tottf * totidf
        element_square = element_square + (element_weight ** 2)

    norm_weight = math.sqrt(element_square)
    return norm_weight

#Function to get the weight of token in a document
def getweight(name_of_doc, name_of_token):
    f_tf = gettf(name_of_doc, name_of_token)
    f_idf = getidf(name_of_token)
    if (f_tf > 0):
        f_tf = f_tf / norm_weight(name_of_doc, name_of_token)
        weight = f_tf * f_idf
    else:
        weight = 0
    return weight

def norm_tq(queryFreq):
    sumof_q = 0
    for element,cost in queryFreq.items():
        sumof_q = sumof_q + (cost ** 2)
    sum_norm = math.sqrt(sumof_q)
    return sum_norm


#Function to find the ranking of document
def query(queryString):
    tokens = tokenizer.tokenize(queryString)
    element_q = []
    sine_query_doc = 0
    q_count = 0
    name_of_document = None

    for t in tokens:
        if t not in words_in_stop:
            element_q.append(stemmer.stem(t))

    queryFreq = dict.fromkeys(element_q, 0)

    for element,cost in queryFreq.items():
        for t in tokens:
            if element in t:
                q_count = q_count + 1
        queryFreq[element] = q_count

    norm_qw = norm_tq(queryFreq)

    for key,value in document.items():
        sum = 0
        for element,cost in queryFreq.items():
            tf_tq = 1 + math.log10(cost)
            w_tq = tf_tq / norm_qw
            w_td = getweight(key, element)
            sum = sum + (w_tq * w_td)

        if (sine_query_doc < sum):
            sine_query_doc = sum
            name_of_document = key
    return (name_of_document, sine_query_doc)

main()