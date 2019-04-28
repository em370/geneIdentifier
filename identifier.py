def features(sequence, i):
    yield "word=" + sequence[i].lower()
    if sequence[i].isupper():
        yield "Uppercase"
        
        
        
from seqlearn.datasets import load_conll
X_train, y_train, lengths_train = load_conll("gene-trainF18.txt", features)

from seqlearn.perceptron import StructuredPerceptron
clf = StructuredPerceptron()
clf.fit(X_train, y_train, lengths_train)


X_test, y_test, lengths_test = load_conll("gene-trainF18.txt", features)
from seqlearn.evaluation import bio_f_score
y_pred = clf.predict(X_test, lengths_test)
print(bio_f_score(y_test, y_pred))