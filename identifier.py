import nltk
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB

sequence_to_pos_tags = {}



def test2Conll(filename):
	with open(filename,'r') as input:
		with open('Conll'+filename, 'w') as output:  
			for line in input:
				words = line.split()
				if len(words)>0:
					output.write("{}	{}	{}\n".format(words[0],words[1],"F"))
				else:
					output.write("\n");
	return 'Conll'+filename

def parse_word_from_component(component):
    return component.split("\t")[1]

def sentence_from_seq(seq):
    words = []
    for component in seq:
        words.append(component.split("\t")[1])
    return ' '.join(words)               

def get_surrounding_words_pos(sequence, i):
    parts_of_speech = {}
    words = {}
    alpha = {} #checks if its only letters
    pos_tags = sequence_to_pos_tags[sequence]
    if i > 0:
        parts_of_speech['pos_prev'] = pos_tags[i - 1][1]
        words['word_prev'] = parse_word_from_component(sequence[i - 1])
        
    if i < len(sequence) - 1:
        parts_of_speech['pos_next'] = pos_tags[i + 1][1]
        words['word_next'] = parse_word_from_component(sequence[i + 1])
    if i>1:
        parts_of_speech['pos_pprev'] = pos_tags[i - 2][1]
        words['word_pprev'] = parse_word_from_component(sequence[i - 2])
    if i < len(sequence) - 2:
        parts_of_speech['pos_nnext'] = pos_tags[i + 2][1]
        words['word_nnext'] = parse_word_from_component(sequence[i + 2])
        
    return words, parts_of_speech
    

def features(sequence, i):
    word = parse_word_from_component(sequence[i].lower())
    yield "word=" + word
    yield "al=" + str(word.isalpha())
    if sequence[i].isupper():
        yield "Case= Upper"
    elif sequence[i].islower():
        yield "Case= Lower"
    else:
        yield "Case= Mix"
    if sequence in sequence_to_pos_tags:
        pos_tags = sequence_to_pos_tags[sequence]
        pos = pos_tags[i][1]
    else:
        sentence = sentence_from_seq(sequence)
        tokens = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)
        sequence_to_pos_tags[sequence] = pos_tags
        pos = pos_tags[i][1]
    yield "pos_curr:{}".format(pos)
    words, pos = get_surrounding_words_pos(sequence, i)
    if 'word_prev' in words:
        yield "word_prev:{}".format(words['word_prev'])
    if 'pos_prev' in pos:
        yield "pos_prev:{}".format(pos['pos_prev'])
    if 'word_next' in words:
        yield "word_next:{}".format(words['word_next'])
    if 'pos_next' in pos:
        yield "pos_next:{}".format(pos['pos_next'])
    if 'pos_nnext' in pos:
        yield "pos_nnext:{}".format(pos['pos_nnext'])
    if 'word_nnext' in words:
        yield "word_nnext:{}".format(words['word_nnext'])
    if 'word_pprev' in words:
        yield "word_pprev:{}".format(words['word_pprev'])
    if 'pos_pprev' in pos:
        yield "pos_pprev:{}".format(pos['pos_pprev'])

        
        
        
import os
from seqlearn.datasets import load_conll

testFile = "test-run-test.txt"
testFileAnswers = "test-run-test-with-keys.txt"
trainFile = "gene-trainF18.txt"#"train.txt"
notConll = 1 #set this to 1 if the test file doesn't containt tags

if notConll == 1:
	conllFile = test2Conll(testFile)	
else:
	conllFile = inputFile	
	
	
X_train, y_train, lengths_train = load_conll(trainFile, features)  #train.txt
#print(X_train[1])
from seqlearn.perceptron import StructuredPerceptron
clf = StructuredPerceptron()
clf.fit(X_train, y_train, lengths_train)

X_test, y_test, lengths_test = load_conll(conllFile, features) #eval.txt
X_ans, y_ans, lengths_ans = load_conll(testFileAnswers, features)

from seqlearn.evaluation import bio_f_score
y_pred = clf.predict(X_test, lengths_test)

print(bio_f_score(y_ans, y_pred))
wordsNTags = list(zip(X_test,y_pred));
#print(list(zip(X_train,y_pred)))


my_numbers = []
my_words = []
prevNum = 0

with open(testFile,'r') as f:
    for line in f:
        words = line.split()
        if len(words)>0:
            my_numbers.append(words[0])
            my_words.append(words[1])
            
with open('output.txt', 'w') as f:            
    for number,word,tag in zip(my_numbers,my_words,y_pred):
        if prevNum < int(number):
            f.write("{}	{}	{}\n".format(number,word,tag))
            prevNum = int(number)
        else:
            f.write("\n")
            f.write("{}	{}	{}\n".format(number,word,tag))
            prevNum = int(number)

print(bio_f_score(y_test, y_pred))

os.system("python evalNER.py "+testFileAnswers+ " output.txt")



	

