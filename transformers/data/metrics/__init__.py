from ...file_utils import is_sklearn_available, requires_sklearn
if is_sklearn_available():
    from sklearn.metrics import f1_score, matthews_corrcoef
    from scipy.stats import pearsonr, spearmanr

def simple_accuracy(preds, labels):
    requires_sklearn(simple_accuracy)
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.__init__.acc_and_f1', 'acc_and_f1(preds, labels)', {'requires_sklearn': requires_sklearn, 'acc_and_f1': acc_and_f1, 'simple_accuracy': simple_accuracy, 'f1_score': f1_score, 'preds': preds, 'labels': labels}, 1)

def pearson_and_spearman(preds, labels):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.__init__.pearson_and_spearman', 'pearson_and_spearman(preds, labels)', {'requires_sklearn': requires_sklearn, 'pearson_and_spearman': pearson_and_spearman, 'pearsonr': pearsonr, 'spearmanr': spearmanr, 'preds': preds, 'labels': labels}, 1)

def glue_compute_metrics(task_name, preds, labels):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.__init__.glue_compute_metrics', 'glue_compute_metrics(task_name, preds, labels)', {'requires_sklearn': requires_sklearn, 'glue_compute_metrics': glue_compute_metrics, 'matthews_corrcoef': matthews_corrcoef, 'simple_accuracy': simple_accuracy, 'acc_and_f1': acc_and_f1, 'pearson_and_spearman': pearson_and_spearman, 'task_name': task_name, 'preds': preds, 'labels': labels}, 1)

def xnli_compute_metrics(task_name, preds, labels):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.__init__.xnli_compute_metrics', 'xnli_compute_metrics(task_name, preds, labels)', {'requires_sklearn': requires_sklearn, 'xnli_compute_metrics': xnli_compute_metrics, 'simple_accuracy': simple_accuracy, 'task_name': task_name, 'preds': preds, 'labels': labels}, 1)

