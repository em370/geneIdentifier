import nltk
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB

sequence_to_pos_tags = {}

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
    pos_tags = sequence_to_pos_tags[sequence]
    if i > 0:
        parts_of_speech['pos_prev'] = pos_tags[i - 1][1]
        words['word_prev'] = parse_word_from_component(sequence[i - 1])
    if i < len(sequence) - 1:
        parts_of_speech['pos_next'] = pos_tags[i + 1][1]
        words['word_prev'] = parse_word_from_component(sequence[i - 1])
    return words, parts_of_speech
    

def features(sequence, i):
    yield "word=" + parse_word_from_component(sequence[i].lower())
    if sequence[i].isupper():
        yield "Uppercase"
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
        
        
        
from seqlearn.datasets import load_conll
X_train, y_train, lengths_train = load_conll("train.txt", features)

from seqlearn.perceptron import StructuredPerceptron
clf = StructuredPerceptron()
clf.fit(X_train, y_train, lengths_train)

X_test, y_test, lengths_test = load_conll("eval.txt", features)
from seqlearn.evaluation import bio_f_score
y_pred = clf.predict(X_test, lengths_test)

print(bio_f_score(y_test, y_pred))
