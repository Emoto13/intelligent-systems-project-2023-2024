import knn 
import naive_bayes_binomial 
import naive_bayes_multinomial
import decision_tree
import matplotlib.pyplot as plt 
import numpy as np 
from utils import get_classes

def get_class_plot_by_metric(model_to_metrics, metric_name='Accuracy', field_to_use='micro_accuracy'):

    x = np.arange(len(get_classes()))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')


    for model, metrics in model_to_metrics.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, getattr(metrics, field_to_use), width, label=model)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.\
    ax.set_ylabel(metric_name)
    ax.set_title(f'Models {metric_name.lower()} by classes')
    ax.set_xticks(x + 0.25, get_classes())
    ax.legend(loc='upper left', ncols=len(get_classes()))
    plt.show()

if __name__ == '__main__':
    models_to_cb = { 
        "naive bayes binomial": naive_bayes_binomial.train_test_evaluate,
        "naive bayes multinomial": naive_bayes_multinomial.train_test_evaluate,
        "kNN": knn.train_test_evaluate,
        "decision tree": decision_tree.train_test_evaluate,
    }

    model_to_metrics = {}
    for model_name, cb in models_to_cb.items():
        metrics = cb()
        model_to_metrics[model_name] = metrics
        print(model_name, metrics)

    models = list(models_to_cb.keys())

    metric_to_attr = {
        'Accuracy': 'micro_accuracy',
        'Precision': 'micro_precision',
        'Recall': 'micro_recall',
        'F1 Score': 'micro_f1',
    }
    for metric_name, attr in metric_to_attr.items():
        get_class_plot_by_metric(model_to_metrics, metric_name=metric_name, field_to_use=attr)

    for model in models:
        plt.matshow(model_to_metrics[model].confusion_matrix)
        plt.show()

        x = np.arange(len(get_classes()))  # the label locations
        width = 0.2  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for model, metrics in model_to_metrics.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, [metrics.accuracy, metrics.macro_precision, metrics.macro_recall, metrics.macro_f1], width, label=model)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.\
        ax.set_ylabel('Macro')
        ax.set_title(f'Models macro metrics')
        ax.set_xticks(x + 0.25, ['Accuracy', 'Precision', 'Recall', 'F1 Score'])
        ax.legend(loc='upper left', ncols=len(get_classes()))
        plt.show()

    