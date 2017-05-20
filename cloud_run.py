
# coding: utf-8

# # Sentence Retrieval

# In[123]:

import json
import math
import nltk
from time import sleep
from tqdm import tqdm

train_file = open("data/train.json",'r')

# dev  ----> test
dev_file = open("data/dev.json",'r')
test_file=open("data/test.json",'r')
train = json.loads(train_file.read())
dev = json.loads(dev_file.read())
test = json.loads(test_file.read())


# In[124]:

from nltk.corpus import stopwords
# may be stem?

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
stopword =  stopwords.words()

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma

def doc_word_dict(doc):
    word_dict = set()
    for sent in doc['sentences']:
        for word in  word_tokenizer.tokenize(sent):
            word = lemmatize(word.lower())
            if word not in stopword:
                word_dict.add(word)
    return word_dict


# In[125]:

def get_BOW(sent):
    term_dict={}
    for word in word_tokenizer.tokenize(sent):
        word = lemmatize(word.lower())
        if word not in stopword:
            term_dict[word]=term_dict.setdefault(word,0)+1
    return term_dict

def cal_BOW(doc):
    doc_term_matrix = [] 
    for sent in doc['sentences']:
        temp = get_BOW(sent)
        doc_term_matrix.append(temp)
    return doc_term_matrix


# In[126]:

def get_best_doc_num2(query):
    query =  transformer.transform(vectorizer.transform(get_BOW(query)))
    result={}
    for x in range(term_matrix.shape[0]):
         result[x]=cos_distance(query.toarray(),term_matrix[x].toarray())
            
    minvalue=1
    first=0
    for item in result:
        if minvalue > result[item]:
            minvalue=result[item]
            first=item     
    del result[first]
    
    minvalue=1
    second=0
    for item in result:
        if minvalue > result[item]:
            minvalue=result[item]
            second=item     
    return first,second


# In[127]:

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial.distance import cosine as cos_distance

vectorizer = DictVectorizer()
transformer = TfidfTransformer(smooth_idf=False,norm=None)

match_sent= []
count = 0

for dev_doc in tqdm(dev, desc='Extracting sentences from documents'):
    count += 1
    doc_match_sent = []
    term_matrix = transformer.fit_transform(vectorizer.fit_transform(cal_BOW(dev_doc)))
    for qa in dev_doc['qa']:
        doc_match_sent.append(get_best_doc_num2(qa['question']))
    match_sent.append(doc_match_sent)


# # Entity Extraction

# In[128]:

from nltk.tag.stanford import StanfordNERTagger
st = StanfordNERTagger('/home/ec2-user/stanford-ner-2016-10-31/classifiers/english.muc.7class.distsim.crf.ser.gz',
               '/home/ec2-user/stanford-ner-2016-10-31/stanford-ner.jar') 

test_tag = []
for i in range(len(match_sent)):
    test_sent_tag = []
    for j,k in match_sent[i]:
        test_sent_tag.append(word_tokenizer.tokenize(dev[i]['sentences'][j] + ' ' + dev[i]['sentences'][k]))
    test_sent_tag = st.tag_sents(test_sent_tag)
    test_tag.append(test_sent_tag)


# In[129]:

def tune_other(tag_list):
    for i in range(len(tag_list)):
        for j in range(len(tag_list[i])):
            for k in range(len(tag_list[i][j])):
                term,tag = tag_list[i][j][k]
                if term!='' and (tag == "ORGANIZATION"  or (len(term)>0 and (term,tag)!=tag_list[i][j][0] and tag == 'O' and term[0].isupper())):
                    tag_list[i][j][k] = (term,"OTHER")

tune_other(test_tag)


# In[130]:

def combine_entity(tag_list):
    for k in range(len(tag_list)):
        for i in range(len(tag_list[k])):
            j = 0
            while j < len(tag_list[k][i])-2:
                term,tag = tag_list[k][i][j]
                term_n,tag_n = tag_list[k][i][j+1]
                if tag == tag_n and tag != "O":
                    temp =  (term + " " + term_n,tag)
                    tag_list[k][i][j] = temp
                    del tag_list[k][i][j+1]
                j += 1

combine_entity(test_tag)


# In[131]:

test_list = []
for k in test_tag:
    test_doc_list = []
    for i in k:
        test_ = []
        for term,tag in i:
            if term != '':
                test_.append(term)
        test_doc_list.append(test_)
    test_list.append(test_doc_list)


# In[132]:

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

tags = ["PERSON","LOCATION","NUMBER","OTHER"]
entity_pool = []
for k in tqdm(test_tag, desc='Extracting entities'):
    entity_doc_pool = []
    for i in k:
        sent_tag_dict = dict.fromkeys(tags,[])
        for j in range(len(i)):
            term,tag = i[j]
            if tag == "PERSON" or tag == "LOCATION" or tag == "OTHER":
                sent_tag_dict[tag] = sent_tag_dict[tag]+ [term]
            elif tag == "DATE" or tag == "TIME" or tag == "PERCENT" or hasNumbers(term):
                sent_tag_dict["NUMBER"] = sent_tag_dict["NUMBER"]+ [term]
        entity_doc_pool.append(sent_tag_dict)
    entity_pool.append(entity_doc_pool)


# # Answer Ranking

# In[138]:

import operator
import nltk

# A simple rule-based question type classifier based on key words 
def get_question_type(question):
    # TODO: HAND-CODED, NEED TO BE REFINED!!
    # TODO: need to low-case to compare?
    
    type_rules = [
        ('PERSON', ["Who", "Whose", "Whom", "whom"]),
        ('LOCATION', ["Where"]),
        ('NUMBER', ["When", "few", "little", "much", "many","size",
                   "young", "old", "long", "year", "years", "day"])
    ]
    
    q_type = None
    for question_type, key_words in type_rules:
        if q_type == None:
            for key_word in key_words:
                if key_word in question:
                    q_type = question_type
                    break
    if q_type == None:
        q_type = 'OTHER'
    # log for error analysis
    output_file.write('Q_type: ' + '\t' + q_type + '\n')
    return q_type


# In[139]:

from nltk import word_tokenize

# among entities of the same type, the prefered entity should be 
# the one which is closer in the sentence to a open-class word
# from the question.
# ----> nouns, verbs, adjectives, and adverbs.
def get_preferred_entity(entity_list, sentence, question):
    preferred_entity = None
    sentence_text = sentence
    question_text = word_tokenize(question)
    sentence_tag = nltk.pos_tag(sentence_text,tagset='universal')
    question_tag = nltk.pos_tag(question_text,tagset='universal')
    
    # initialize a list for comparing, and set all elements as 0
    is_open_word = [0] * len(sentence_text)
    # find an open word in the question
    for word, tag in question_tag:
        if tag in ['ADJ','NOUN','VERB','ADV']:
            # if the open word appears in the sentence, then mark as 1
            for i in range(len(sentence_text)):
                if sentence_text[i] == word:
                    is_open_word[i] = 1
    
    # find the closest distance to an open-class word for an entity
    def get_distance(entity):
        # get the position of entity, and find the open class word 
        # from the nearest at both sides
        distance = None
        position = sentence_text.index(entity)
        for i in range(1, len(sentence_text)):
            if position - i >= 0:
                if is_open_word[position - i] == 1:  # find an open-class word on the left
                    distance = i
                    break
                elif position + i < len(is_open_word):  # find an open-class word on the right
                    if is_open_word[position + i] == 1:
                        distance = i
                        break
                else:
                    distance = len(sentence_text) + 1  # didn't find open-class words
        return distance
    
    # get distance for each entity and choose the best one
    all_distance = []
    for entity in entity_list:
        all_distance.append(get_distance(entity))
        preferred_entity = entity_list[all_distance.index(min(all_distance))]

    return preferred_entity
    


# In[140]:

def get_ranked_ans(entities_dic, question, sentence):
    # identify if the entity set is empty. If True, return nothing
    is_empty = True
    for values in entities_dic.values():
        if len(values) != 0:
            is_empty = False
            
    if is_empty == False:
        q_type = get_question_type(question)
        tmp_rank = {}
        for ent_type,entities in entities_dic.items():
            # answers whose content words all appear in the question should be ranked lowest.
            for entity in entities:
                tmp_rank[entity] = tmp_rank.setdefault(entity,0)
                if entity in question:
                    tmp_rank[entity] = tmp_rank.setdefault(entity,0) - 1
            # Answers which match the question type should be ranked higher than those that don't
            if ent_type == q_type and ent_type != 'OTHER':
                for entity in entities:
                    tmp_rank[entity] = tmp_rank.setdefault(entity,0) + 1
                ######## TODO: Apply this to all types?
            # entity closer in the sentence to a closed-class word should be preferred
            preferred_entity = get_preferred_entity(entities, sentence, question)
            if preferred_entity != None:
                tmp_rank[preferred_entity] = tmp_rank.setdefault(preferred_entity,0) + 1
        # sort and choose the best answer
        sorted_ans = sorted(tmp_rank.items(), key=operator.itemgetter(1), reverse=True)
        # print sorted_ans
        # log for error analysis
        output_file.write('Ranked Answers: ' + '\t' + str(sorted_ans).encode('utf-8') + '\n\n')
        # TODO: bug here. list out of index??? why?
        if len(sorted_ans) != 0:
            best_ans = sorted_ans[0][0]
        else:
            best_ans = ''
        return best_ans
    else:
        return ''


# In[141]:

num = 0
count = 1
correct_sum = 0
corr_sen_retr_count = 0

with open("result.txt",'w') as output_file:
    for i in tqdm(range(len(match_sent)), desc='Answering'):
        for j in range(len(match_sent[i])):
            result = get_ranked_ans(entity_pool[i][j], dev[i]["qa"][j]['question'], test_list[i][j])
            output_file.write('Retrieved Entities: ' + '\t' + str(entity_pool[i][j]) + '\n\n')
#             q_id = dev[i]["qa"][j]['id']
            count += 1
            cor_answer = dev[i]["qa"][j]['answer']
            Q = dev[i]["qa"][j]['question']
            A_sentence = dev[i]["sentences"][dev[i]["qa"][j]['answer_sentence']]
            sent_1, sent_2 = match_sent[i][j]
            guessed_sentence = dev[i]['sentences'][sent_1] + ' ' + dev[i]['sentences'][sent_2]
            
            if result == cor_answer:
                correct_sum += 1
            else:
                string1 = 'Retrieved Sentence: ' + '\t' + guessed_sentence.encode('utf-8')+"\n\n"
                string1_1 = '==== WRONG SENTENCES! ==== \n' + 'Guessed_Sentence: ' + '\t' + guessed_sentence.encode('utf-8')+"\n\n"
                string1_2 = 'CORRECT_Sentence: ' + '\t' + A_sentence.encode('utf-8')+"\n\n"
                string2 = 'Q: ' + '\t' + Q.encode('utf-8') + '\n\n'
                string3 = 'CORRECT_ANSWER: ' + '\t' + cor_answer.encode('utf-8') + '\n'
                string4 = 'GUESSED_ANSWER: ' + '\t' + result.encode('utf-8')+"\n"
                
                if A_sentence not in guessed_sentence:
                    output_file.write(string1_1)
                    output_file.write(string1_2)
                else:
                    corr_sen_retr_count += 1
                    output_file.write(string1)
                output_file.write(string2)
                output_file.write('='*60 + '\n')
                output_file.write(string3)
                output_file.write(string4)
                output_file.write('='*60 + '\n\n')
    print 'correct sum: ' + str(correct_sum)
    print 'Sentence Recall: ' + str((corr_sen_retr_count+0.0)/count)
    
for i in dev:
    for j in i["qa"]:
        num += 1
print (correct_sum+0.0)/num


# In[142]:

# # run on test data

# with open("result.txt",'w') as output_file:
#     output_file.write('id,answer'+'\n')
#     for i in tqdm(range(len(match_sent)), desc='Answering'):
#         for j in range(len(match_sent[i])):
#             result = get_ranked_ans(entity_pool[i][j], dev[i]["qa"][j]['question'], test_list[i][j])
#             q_id = dev[i]["qa"][j]['id']
#             output_file.write(str(q_id) + ',' + str(result.encode('utf-8')) + '\n')

