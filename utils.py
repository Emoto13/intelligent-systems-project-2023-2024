from nltk.corpus import PlaintextCorpusReader
from dataclasses import dataclass
from typing import Any

def read_corpus(root_path='JOURNALISM.BG/C-MassMedia'):
    return PlaintextCorpusReader(root_path, '.*\.txt')    

def get_classes():
    corpus = read_corpus()
    file_names = corpus.fileids()
    classes_set = set( [ file[:file.find('/')] for file in file_names ] )
    classes = set(classes_set) - set(['Z','D-Society'])
    return classes


def test_classifier(test_class_corpus, gamma = None, predicted = [], actual = []):
    L = { c:  len(v) for c, v in test_class_corpus.items() }

    confusionMatrix = { c: { c : 0 for c in test_class_corpus.keys() } for c in test_class_corpus.keys() } 
    right = 0
    for c, texts in test_class_corpus.items():
        for text in texts:
            c_MAP = gamma(text)
            confusionMatrix[c][c_MAP] += 1
            right += 1 if c_MAP == c else 0
    
    accuracy = {}
    precision = {}
    recall = {}
    Fscore = {}
    for c in test_class_corpus.keys():
        extracted = sum(confusionMatrix[x][c] for x in test_class_corpus.keys())
        if confusionMatrix[c][c] == 0:
            accuracy[c] = 0.0
            precision[c] = 0.0
            recall[c] = 0.0
            Fscore[c] = 0.0
        else:
            accuracy[c] = confusionMatrix[c][c] / L[c]
            precision[c] = confusionMatrix[c][c] / extracted
            recall[c] = ( confusionMatrix[c][c] / L[c] )
            Fscore[c] = ((2.0 * precision[c] * recall[c]) / (precision[c] + recall[c]))

    A = sum( L[c] * accuracy[c] for c in test_class_corpus.keys() ) / sum(L.values())
    P = sum( L[c] * precision[c] / sum(L.values()) for c in test_class_corpus.keys() )
    R = sum( L[c] * recall[c] / sum(L.values()) for c in test_class_corpus.keys() )
    F1 = (2 * P * R) / (P + R)
    return accuracy, confusionMatrix, precision, recall, Fscore, P, R, F1, A

def display_model_metrics(test_data, gamma):
    accuracy, confusionMatrix, precision, recall, Fscore, P, R, F1, A = test_classifier(test_data, gamma)
    print('Матрица на обърквания: ')
    for row in confusionMatrix.values():
        for val in row.values():
            print('{:4}'.format(val), end = '')
        print()
    print('Toчност:', accuracy)
    print('Прецизност: ', precision)
    print('Обхват: ', recall)
    print('F-оценка: ', Fscore)
    print('Обща точност: ', A ,'прецизност: ', P,', обхват: ', R, ', F-оценка: ', F1)
    print()

@dataclass
class Metrics:
    accuracy: float
    micro_accuracy: Any
    confusion_matrix: Any
    micro_precision: Any
    macro_precision: float
    micro_recall: Any
    macro_recall: float
    micro_f1: Any
    macro_f1: float
