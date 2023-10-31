""" Very heavily inspired by the official evaluation script for SQuAD version 2.0 which was
modified by XLNet authors to update `find_best_threshold` scripts for SQuAD V2.0

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""

import collections
import json
import math
import re
import string
from transformers.tokenization_bert import BasicTokenizer
from ...utils import logging
logger = logging.get_logger(__name__)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.normalize_answer', 'normalize_answer(s)', {'re': re, 'string': string, 's': s}, 1)

def get_tokens(s):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.get_tokens', 'get_tokens(s)', {'normalize_answer': normalize_answer, 's': s}, 1)

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.compute_f1', 'compute_f1(a_gold, a_pred)', {'get_tokens': get_tokens, 'collections': collections, 'a_gold': a_gold, 'a_pred': a_pred}, 1)

def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.get_raw_scores', 'get_raw_scores(examples, preds)', {'normalize_answer': normalize_answer, 'compute_exact': compute_exact, 'compute_f1': compute_f1, 'examples': examples, 'preds': preds}, 2)

def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.apply_no_ans_threshold', 'apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh)', {'scores': scores, 'na_probs': na_probs, 'qid_to_has_ans': qid_to_has_ans, 'na_prob_thresh': na_prob_thresh}, 1)

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.make_eval_dict', 'make_eval_dict(exact_scores, f1_scores, qid_list=None)', {'collections': collections, 'exact_scores': exact_scores, 'f1_scores': f1_scores, 'qid_list': qid_list}, 1)

def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]

def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.find_best_thresh_v2', 'find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans)', {'preds': preds, 'scores': scores, 'na_probs': na_probs, 'qid_to_has_ans': qid_to_has_ans}, 3)

def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.find_all_best_thresh_v2', 'find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans)', {'find_best_thresh_v2': find_best_thresh_v2, 'main_eval': main_eval, 'preds': preds, 'exact_raw': exact_raw, 'f1_raw': f1_raw, 'na_probs': na_probs, 'qid_to_has_ans': qid_to_has_ans}, 0)

def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.find_best_thresh', 'find_best_thresh(preds, scores, na_probs, qid_to_has_ans)', {'preds': preds, 'scores': scores, 'na_probs': na_probs, 'qid_to_has_ans': qid_to_has_ans}, 2)

def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.find_all_best_thresh', 'find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans)', {'find_best_thresh': find_best_thresh, 'main_eval': main_eval, 'preds': preds, 'exact_raw': exact_raw, 'f1_raw': f1_raw, 'na_probs': na_probs, 'qid_to_has_ans': qid_to_has_ans}, 0)

def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.squad_evaluate', 'squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0)', {'get_raw_scores': get_raw_scores, 'apply_no_ans_threshold': apply_no_ans_threshold, 'make_eval_dict': make_eval_dict, 'merge_eval': merge_eval, 'find_all_best_thresh': find_all_best_thresh, 'examples': examples, 'preds': preds, 'no_answer_probs': no_answer_probs, 'no_answer_probability_threshold': no_answer_probability_threshold}, 1)

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.get_final_text', 'get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False)', {'collections': collections, 'BasicTokenizer': BasicTokenizer, 'logger': logger, 'pred_text': pred_text, 'orig_text': orig_text, 'do_lower_case': do_lower_case, 'verbose_logging': verbose_logging}, 2)

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics._get_best_indexes', '_get_best_indexes(logits, n_best_size)', {'logits': logits, 'n_best_size': n_best_size}, 1)

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics._compute_softmax', '_compute_softmax(scores)', {'math': math, 'scores': scores}, 1)

def compute_predictions_logits(all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case, output_prediction_file, output_nbest_file, output_null_log_odds_file, verbose_logging, version_2_with_negative, null_score_diff_threshold, tokenizer):
    """Write final predictions to the json file and log-odds of null if needed."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.compute_predictions_logits', 'compute_predictions_logits(all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case, output_prediction_file, output_nbest_file, output_null_log_odds_file, verbose_logging, version_2_with_negative, null_score_diff_threshold, tokenizer)', {'logger': logger, 'collections': collections, '_get_best_indexes': _get_best_indexes, 'get_final_text': get_final_text, '_compute_softmax': _compute_softmax, 'json': json, 'all_examples': all_examples, 'all_features': all_features, 'all_results': all_results, 'n_best_size': n_best_size, 'max_answer_length': max_answer_length, 'do_lower_case': do_lower_case, 'output_prediction_file': output_prediction_file, 'output_nbest_file': output_nbest_file, 'output_null_log_odds_file': output_null_log_odds_file, 'verbose_logging': verbose_logging, 'version_2_with_negative': version_2_with_negative, 'null_score_diff_threshold': null_score_diff_threshold, 'tokenizer': tokenizer}, 1)

def compute_predictions_log_probs(all_examples, all_features, all_results, n_best_size, max_answer_length, output_prediction_file, output_nbest_file, output_null_log_odds_file, start_n_top, end_n_top, version_2_with_negative, tokenizer, verbose_logging):
    """XLNet write prediction logic (more complex than Bert's).
    Write final predictions to the json file and log-odds of null if needed.

    Requires utils_squad_evaluate.py
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.data.metrics.squad_metrics.compute_predictions_log_probs', 'compute_predictions_log_probs(all_examples, all_features, all_results, n_best_size, max_answer_length, output_prediction_file, output_nbest_file, output_null_log_odds_file, start_n_top, end_n_top, version_2_with_negative, tokenizer, verbose_logging)', {'collections': collections, 'logger': logger, 'get_final_text': get_final_text, '_compute_softmax': _compute_softmax, 'json': json, 'all_examples': all_examples, 'all_features': all_features, 'all_results': all_results, 'n_best_size': n_best_size, 'max_answer_length': max_answer_length, 'output_prediction_file': output_prediction_file, 'output_nbest_file': output_nbest_file, 'output_null_log_odds_file': output_null_log_odds_file, 'start_n_top': start_n_top, 'end_n_top': end_n_top, 'version_2_with_negative': version_2_with_negative, 'tokenizer': tokenizer, 'verbose_logging': verbose_logging}, 1)

