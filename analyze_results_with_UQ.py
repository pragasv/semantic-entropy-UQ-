"""Compute overall performance metrics from predicted uncertainties."""
import argparse
import functools
import logging
import os
import pickle

import json

import numpy as np
from scipy.optimize import minimize_scalar
# import wandb

from uncertainty.utils import utils
from uncertainty.utils.eval_utils import (
    bootstrap, compatible_bootstrap, auroc, accuracy_at_quantile,
    area_under_thresholded_accuracy)


utils.setup_logger()

result_dict = {}

SMALL_RUN = True
SMALL_SET_SIZE = 250
LAMBDA_VAL_LIST = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, -0.000001, -0.00001, -0.0001, -0.001, -0.01]

LAMBDA_VAL_LIST_ENTROPY_MAX_APPROACH = [5, 15, 25, 50, 75, 100] 


if SMALL_RUN==True:
    UNC_MEAS = f'uncertainty_measures_max_idx_{SMALL_SET_SIZE}_withreversemap.pkl'
    RESULTS_FILE = f'analysis_result_max_idx_{SMALL_SET_SIZE}_withreversemap_withUQ.json'
else:
    UNC_MEAS = 'uncertainty_measures.pkl'
    RESULTS_FILE = 'analysis_result_withUQ.json'
    
ROOT_DIR = '' #THIS NEEDS TO BE UPDATED 


def analyze_run_with_UQ(
        wandb_runid, RUN_ID = 'qsy0i0cb', assign_new_wandb_id=False, answer_fractions_mode='default',
        experiment_lot=None, entity=None):
    """Analyze the uncertainty measures for a given wandb run id."""
    # logging.info('Analyzing wandb_runid `%s`.', wandb_runid)

    # Set up evaluation metrics.
    if answer_fractions_mode == 'default':
        answer_fractions = [0.8, 0.9, 0.95, 1.0]
    elif answer_fractions_mode == 'finegrained':
        answer_fractions = [round(i, 3) for i in np.linspace(0, 1, 20+1)]
    else:
        raise ValueError

    rng = np.random.default_rng(41)
    eval_metrics = dict(zip(
        ['AUROC', 'area_under_thresholded_accuracy', 'mean_uncertainty'],
        list(zip(
            [auroc, area_under_thresholded_accuracy, np.mean],
            [compatible_bootstrap, compatible_bootstrap, bootstrap]
        )),
    ))
    for answer_fraction in answer_fractions:
        key = f'accuracy_at_{answer_fraction}_answer_fraction'
        eval_metrics[key] = [
            functools.partial(accuracy_at_quantile, quantile=answer_fraction),
            compatible_bootstrap]

    # Load the results dictionary from a pickle file.
    with open(f'{ROOT_DIR}/{RUN_ID}/files/analysis_praga/{UNC_MEAS}', 'rb') as file:
        results_old = pickle.load(file)
        
    
    dataset_bundle = []
    path = f'{ROOT_DIR}/{RUN_ID}/files/analysis_praga/spinchain_measures_max_idx_250_withreversemap.json'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    dataset_bundle.append((data, 'dummy'))  # Collect for combined plotting


    for LAMBDA_VAL in  LAMBDA_VAL_LIST:
        re_all, re_with_UQ_all = [], []
        
        for key in dataset_bundle[0][0].keys():

            log_liks_agg = dataset_bundle[0][0][key]['log_liks_agg'][0]
            semantic_ids = dataset_bundle[0][0][key]['semantic_idcs'][0]
            UQ_values = dataset_bundle[0][0][key]['KME_UQ_reversemapped'][0]
            
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')

            ### CHANGE(praga)compute: 2nd order Renyis 
            re = predictive_entropy_renyi(log_likelihood_per_semantic_id)
            
            log_likelihood_per_semantic_id, UQ_per_semantic_id = logsumexp_by_id_with_UQ(semantic_ids, log_liks_agg, UQ_values, agg='sum_normalized')
            re_with_UQ = predictive_entropy_renyi_with_UQ(log_likelihood_per_semantic_id, UQ_per_semantic_id, lambda_val=LAMBDA_VAL)
            
            re_all.append(re)
            re_with_UQ_all.append(re_with_UQ)
            
            
        if results_old['uncertainty_measures']['semantic_renyi_entropy'] == re_all:
            results_old['uncertainty_measures'][f'semantic_renyi_entropy_withUQ_{LAMBDA_VAL}'] = re_with_UQ_all
        else:
            print('SREs did not match')
            raise ValueError
        

    
    for LAMBDA_VAL in  LAMBDA_VAL_LIST_ENTROPY_MAX_APPROACH:
        re_all, re_with_UQ_all = [], []
        
        for key in dataset_bundle[0][0].keys():

            log_liks_agg = dataset_bundle[0][0][key]['log_liks_agg'][0]
            semantic_ids = dataset_bundle[0][0][key]['semantic_idcs'][0]
            UQ_values = dataset_bundle[0][0][key]['KME_UQ_reversemapped'][0]
            
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')

            ### CHANGE(praga)compute: 2nd order Renyis 
            re = predictive_entropy_renyi(log_likelihood_per_semantic_id)
            
            log_likelihood_per_semantic_id, UQ_per_semantic_id = logsumexp_by_id_with_UQ(semantic_ids, log_liks_agg, UQ_values, agg='sum_normalized')
            re_with_UQ = predictive_entropy_renyi_with_entropy_maximization(log_likelihood_per_semantic_id, UQ_per_semantic_id, lambda_val=LAMBDA_VAL)
            
            re_all.append(re)
            re_with_UQ_all.append(re_with_UQ)
            
            
        if results_old['uncertainty_measures']['semantic_renyi_entropy'] == re_all:
            results_old['uncertainty_measures'][f'semantic_renyi_entropy_withEntropyMax_{LAMBDA_VAL}'] = re_with_UQ_all
        else:
            print('SREs did not match')
            raise ValueError

    result_dict = {'performance': {}, 'uncertainty': {}}

    # First: Compute simple accuracy metrics for model predictions.
    all_accuracies = dict()
    all_accuracies['accuracy'] = 1 - np.array(results_old['validation_is_false'])

    for name, target in all_accuracies.items():
        result_dict['performance'][name] = {}
        result_dict['performance'][name]['mean'] = np.mean(target)
        result_dict['performance'][name]['bootstrap'] = bootstrap(np.mean, rng)(target)

    rum = results_old['uncertainty_measures']
    if 'p_false' in rum and 'p_false_fixed' not in rum:
        # Restore log probs true: y = 1 - x --> x = 1 - y.
        # Convert to probs --> np.exp(1 - y).
        # Convert to p_false --> 1 - np.exp(1 - y).
        rum['p_false_fixed'] = [1 - np.exp(1 - x) for x in rum['p_false']]

    # Next: Uncertainty Measures.
    # Iterate through the dictionary and compute additional metrics for each measure.
    for measure_name, measure_values in rum.items():
        logging.info('Computing for uncertainty measure `%s`.', measure_name)
        
        if measure_name == 'semantic_renyi_entropy': 
            pass
        if measure_name == 'KME_UQ': 
            pass 

        # Validation accuracy.
        validation_is_falses = [
            results_old['validation_is_false'],
            results_old['validation_unanswerable']
        ]

        logging_names = ['', '_UNANSWERABLE']

        # Iterate over predictions of 'falseness' or 'answerability'.
        for validation_is_false, logging_name in zip(validation_is_falses, logging_names):

            name = measure_name + logging_name
            result_dict['uncertainty'][name] = {}

            validation_is_false = np.array(validation_is_false)
            validation_accuracy = 1 - validation_is_false
            if len(measure_values) > len(validation_is_false):
                # This can happen, but only for p_false.
                if 'p_false' not in measure_name:
                    raise ValueError
                logging.warning(
                    'More measure values for %s than in validation_is_false. Len(measure values): %d, Len(validation_is_false): %d',
                    measure_name, len(measure_values), len(validation_is_false))
                measure_values = measure_values[:len(validation_is_false)]

            fargs = {
                'AUROC': [validation_is_false, measure_values],
                'area_under_thresholded_accuracy': [validation_accuracy, measure_values],
                'mean_uncertainty': [measure_values]}

            for answer_fraction in answer_fractions:
                fargs[f'accuracy_at_{answer_fraction}_answer_fraction'] = [validation_accuracy, measure_values]

            for fname, (function, bs_function) in eval_metrics.items():
                metric_i = function(*fargs[fname])
                result_dict['uncertainty'][name][fname] = {}
                result_dict['uncertainty'][name][fname]['mean'] = metric_i
                logging.info("%s for measure name `%s`: %f", fname, name, metric_i)
                result_dict['uncertainty'][name][fname]['bootstrap'] = bs_function(
                    function, rng)(*fargs[fname])

    # wandb.log(result_dict)
    logging.info(
        'Analysis for wandb_runid `%s` finished. Full results dict: %s',
        wandb_runid, result_dict
    )
    
    ## saving the analysis result 
    with open(f'{ROOT_DIR}/{RUN_ID}/files/analysis/{RESULTS_FILE}', "w") as f:
        json.dump(result_dict, f, indent=4)



def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum_normalized':
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def normalize_uq(uq_values):
    min_val = np.min(uq_values)
    max_val = np.max(uq_values)
    if max_val == min_val:
        return np.zeros_like(uq_values)  # Avoid division by zero
    return (uq_values - min_val) / (max_val - min_val)

def logsumexp_by_id_with_UQ(semantic_ids, log_likelihoods, UQ_values, agg='sum_normalized'):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    UQ_values_normalized = normalize_uq(UQ_values)
    
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []
    UQ_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        id_UQ_values_normalized = [UQ_values_normalized[i] for i in id_indices]
        
        if agg == 'sum_normalized':
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
            
            id_UQ_values_normalized =  np.mean(id_UQ_values_normalized)
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)
        UQ_per_semantic_id.append(id_UQ_values_normalized)

    return log_likelihood_per_semantic_id, UQ_per_semantic_id

def predictive_entropy_renyi(log_probs, order=2):
    entropy = - np.log(np.sum(np.power(np.exp(log_probs), order)))
    return entropy 

def kl_divergence(p_hat, q_i):
    return p_hat * np.log(p_hat / q_i) # + (1 - p_hat) * np.log((1 - p_hat) / (1 - q_i))

def objective(p_hat, q_i, lambda_, uq_adjustment):
    if p_hat <= 0.0 or p_hat >= 1.0:
        return np.inf  # avoid invalid log(0)
    first_term = - np.log(2 * p_hat**2 - (2 * p_hat) + 1)
    # second_term = - lambda_ * uq_adjustment * kl_divergence(p_hat, q_i)
    
    second_term = - lambda_ * (1/(uq_adjustment+1)) * kl_divergence(p_hat, q_i)
    return -(first_term + second_term)  # negative because we use minimize_scalar

def find_optimal_p_hats(log_probs, lambda_, uq_adjustments):
    # Normalize log_probs
    # log_probs = log_probs - np.log(np.sum(np.exp(log_probs)))  # log softmax
    q = np.exp(log_probs)  # get the probabilities

    optimal_p_hats = []

    for q_i, uq_adj_i in zip(q, uq_adjustments):
        res = minimize_scalar(
            objective,
            bounds=(1e-6, 1 - 1e-6),
            args=(q_i, lambda_, uq_adj_i),
            method='bounded'
        )
        optimal_p_hats.append(res.x)

    return np.array(optimal_p_hats)

def predictive_entropy_renyi_with_UQ(log_probs, UQ_per_semantic_id, order=2, lambda_val=0.01):
    UQ_adjustment = lambda_val * np.array(UQ_per_semantic_id)
    entropy_adjusted_uq = - np.log(np.sum(np.power(np.exp(log_probs), order) + UQ_adjustment))
    return  entropy_adjusted_uq 

def predictive_entropy_renyi_with_entropy_maximization(log_probs, UQ_per_semantic_id, order=2, lambda_val=0.01, return_entropy_max_prob=False):
    
    # UQ_adjustment = lambda_val * np.array(UQ_per_semantic_id)
    
    try:
        entropy_maxed_probs = find_optimal_p_hats(log_probs, lambda_val, np.array(UQ_per_semantic_id))
        ## normalize 
        entropy_maxed_probs = entropy_maxed_probs/sum(entropy_maxed_probs)
        
    except:
        raise("optimal p did not converge")
        
    entropy_adjusted_uq = - np.log(np.sum(np.power(entropy_maxed_probs, order)))
    
    if return_entropy_max_prob:
        return  entropy_adjusted_uq, entropy_maxed_probs
    else:
        return  entropy_adjusted_uq 