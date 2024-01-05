from nltk.corpus import PlaintextCorpusReader
import random 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, precision_recall_fscore_support
from utils import Metrics,get_classes

class DataEntry:
    def __init__(self, text, target):
        self.text = text
        self.target = target

class DecisionTree:
    def __init__(self):
        self.count_vec = CountVectorizer(input='filename')
        self.tfidf_transformer = TfidfTransformer()
        self.decision_tree = DecisionTreeClassifier()
   
    def split_data(self, data, labels, test_fraction=0.1):
        test_portion = int(data.shape[0] * test_fraction)
        return data[test_portion:], data[:test_portion], labels[test_portion:], labels[:test_portion]

    def read(self, corpus_root = 'JOURNALISM.BG/C-MassMedia'):
        corpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
        file_names = corpus.fileids()
        random.shuffle(file_names)

        file_names = [x for x in file_names if 'Z' not in x and 'D-Society' not in x] 
        labels = [x[:x.index('/')] for x in file_names]
        full_file_names = [f'./JOURNALISM.BG/C-MassMedia/{x}' for x in file_names]
        data_counts = self.count_vec.fit_transform(full_file_names)
        data_tfidf = self.tfidf_transformer.fit_transform(data_counts)

        train, test, train_labels, test_labels = self.split_data(data_tfidf, labels)
        return train, test, train_labels, test_labels

    def train(self, train, labels):
        self.decision_tree.fit(train, labels)

    def apply(self, data):
        predicted = self.decision_tree.predict(data)
        return predicted

def train_test_evaluate():
    s = DecisionTree()
    train, test, train_labels, test_labels = s.read()
    s.train(train, train_labels)
    test_predictions = s.apply(test)

    micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(test_labels, test_predictions, labels=list(get_classes()))
    return Metrics(
        accuracy_score(test_labels, test_predictions), 
        confusion_matrix(test_labels, test_predictions, normalize="true").diagonal(),
        confusion_matrix(test_labels, test_predictions), 
        micro_precision, 
        precision_score(test_labels, test_predictions, average='macro'),
        micro_recall,
        recall_score(test_labels, test_predictions, average='macro'),
        micro_f1_score,
        f1_score(test_labels, test_predictions, average='macro'),
    ) 