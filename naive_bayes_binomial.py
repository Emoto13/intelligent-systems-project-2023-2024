from utils import read_corpus, display_model_metrics, test_classifier, Metrics, get_classes
import math
import random
import numpy as np


class NaiveBayesBinomial:

    def __init__(self):
        self.data = []
        self.prior = {}
        self.cond_prob = {}
        self.Nc = {} # number of docs in class
        self.Nct = {} # number of docs from each class, in which x term is availabe

    def split_data(self, class_to_texts, test_fraction=0.1):
        train = {cls: [] for cls in class_to_texts.keys()}
        test = {cls: [] for cls in class_to_texts.keys()}
        for cls, texts in class_to_texts.items():
            random.shuffle(texts)
            test_count = int(len(texts) * test_fraction)

            test[cls].extend(texts[:test_count])
            train[cls].extend(texts[test_count:])

        return train, test
    
    def combine_dicts(self, list_of_dicts):
        result = {}
        for dicts in list_of_dicts:
            for cls, texts in dicts.items():
                if cls not in result:
                    result[cls] = []
                result[cls].extend(texts)
        return result


    def train(self, train_data):
        N = sum(len(texts) for texts in train_data.values())
        classes = train_data.keys()
        for cls, texts in train_data.items():
            self.Nc[cls] = len(texts)

        for cls in classes:
            for text in train_data[cls]:
                terms = set([ token.lower() for token in text if token.isalpha() ] )
                for term in terms:
                    if term not in self.Nct:
                        self.Nct[term] = { cls: 0 for cls in classes }                 
                    self.Nct[term][cls] += 1
        
        self.prior = { cls: (self.Nc[cls] / N) for cls in classes}
        for term in self.Nct.keys():
            self.cond_prob[term] = { c: (self.Nct[term][c] + 1) / (self.Nc[c] + 2) for c in self.Nc.keys() } # Replace with laplace smoothing
        return self.cond_prob, self.prior, self.Nct

    def apply(self, text):
        max_score = -10000
        answer = None
        terms = set([ token.lower() for token in text if token.isalpha() ] )
        for cls in self.prior.keys():
            score = math.log(self.prior[cls]) + self.intial_cond_prob[cls]
            for term in terms:
                if term not in self.cond_prob:
                    continue
                score += math.log(self.cond_prob[term][cls] / (1.0 - self.cond_prob[term][cls]))
            if score > max_score:
                max_score = score
                answer = cls
        return answer

    def calc_initial_cond_prob(self):
        classes = self.prior.keys()
        self.intial_cond_prob = { cls: 0.0 for cls in classes}
        for c in classes:
            for t in self.cond_prob:
                self.intial_cond_prob[c] += math.log(1.0 - self.cond_prob[t][c])
        return self.intial_cond_prob

    def test(self, test):
        correct = 0
        total = 0
        for cls, texts in test.items():
            total += len(texts)
            for text in texts:
                result = self.apply(text)
                if result == cls:
                   correct += 1
        return correct / total

def train_test_evaluate():
    s = NaiveBayesBinomial()
    corpus = read_corpus()
    file_names = corpus.fileids()
    classes_set = set( [ file[:file.find('/')] for file in file_names ] )
    classes = sorted(set(classes_set) - set(['Z','D-Society']))

    class_to_texts = { c:  [ corpus.words(file) for file in file_names if file.find(c+'/') == 0 ] for c in classes }
    
    train, test = s.split_data(class_to_texts)

    s.train(train)
    s.calc_initial_cond_prob()

    gamma = lambda text: s.apply(text)
    accuracy, confusion_matrix, precision, recall, Fscore, P, R, F1, A = test_classifier(test, gamma)
    classes = get_classes()
    return Metrics(
        accuracy=A,
        micro_accuracy=list(accuracy.values()),
        confusion_matrix=[[confusion_matrix[cls1][cls2] for cls1 in classes] for cls2 in classes], 
        micro_precision=list(precision.values()),
        macro_precision=P,
        micro_recall=list(recall.values()),
        macro_recall=R,
        micro_f1=list(Fscore.values()),
        macro_f1=F1
    )