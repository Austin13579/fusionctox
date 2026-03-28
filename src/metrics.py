from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score,average_precision_score


def evaluate(label, prob):
    pred_label = (prob >= 0.5).astype(int)

    res = {
        'Accuracy': accuracy_score(label, pred_label),
        'AUROC': roc_auc_score(label, prob),
        'AUPRC': average_precision_score(label, prob),
        'Recall': recall_score(label, pred_label),
        'F1': f1_score(label, pred_label),
        'MCC': matthews_corrcoef(label, pred_label)
    }

    return res