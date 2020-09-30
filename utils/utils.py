from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_curve, auc
from typing import List

def get_metrics(true_labels: List[float], predicted_labels: List[float]):
    """[Compute eval metrics from the predictions and ground truths]
    
    Arguments:
        true_labels {[array-like]} -- [ground truth labels]
        predicted_labels {[array-like]} -- [predicted labels]
        verbose: {[bool]} -- [if True print the metrics]
    """
    acc = np.round(metrics.accuracy_score(true_labels,
                                        predicted_labels),
            4)
    precision = np.round(metrics.precision_score(true_labels,
                                    predicted_labels,
                                    average='weighted'),
            4)        
    recall = np.round(
            metrics.recall_score(true_labels,
                                predicted_labels,
                                average='weighted'),
            4)
    f1 = np.round(
            metrics.f1_score(true_labels,
                            predicted_labels,
                            average='weighted'),
            4)
    print('Accuracy:', acc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)
    
    return acc, precision, recall, f1

def display_classification_report(true_labels: List[int], predicted_labels: List[int], 
                                classes: List[int]=[1, 0], dict:bool=False):
    """[Show classification report]
    
    Arguments:
        true_labels {[array-like]} -- [Ground truth labels]
        predicted_labels {[array-like]} -- [Predictions]
    
    Keyword Arguments:
        classes {list} -- [label encoded classes] (default: {[1, 0]})
    """
    if dict:    
        report = metrics.classification_report(y_true=true_labels,
                                                y_pred=predicted_labels,
                                                labels=classes,
                                                output_dict=True)
    else:
        report = metrics.classification_report(y_true=true_labels,
                                                y_pred=predicted_labels,
                                                labels=classes)
        print(report)
    return report



