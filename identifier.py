def features(sequence, i):
    yield "word: " + sequence[i].lower()
    yield "pWord: "+ sequence[i-1].lower()
    #yield "nWord: "+ sequence[i+1].lower()
    #yield sequence[i-1].lower()

    #yield "word=" + sequence[i].lower()
    #if sequence[i].isupper():
    #    yield "Uppercase"
    #else:
    #    yield "Lowercase"
        
import os
from seqlearn.datasets import load_conll
X_train, y_train, lengths_train = load_conll("train.txt", features)
print(X_train[1])
from seqlearn.perceptron import StructuredPerceptron
clf = StructuredPerceptron()
clf.fit(X_train, y_train, lengths_train)


X_test, y_test, lengths_test = load_conll("eval.txt", features)
from seqlearn.evaluation import bio_f_score
y_pred = clf.predict(X_test, lengths_test)
wordsNTags = list(zip(X_test,y_pred));
#print(list(zip(X_train,y_pred)))


my_numbers = []
my_words = []
prevNum = 0

with open('eval.txt','r') as f:
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
            prevNum = int(number)



#with open('output.txt', 'w') as f:
#    for word,tag in wordsNTags:
#        f.write("{} {} \n".format(word,tag))


print(bio_f_score(y_test, y_pred))

os.system("python evalNER.py eval.txt output.txt")