"""Compute overall performance metrics from predicted uncertainties."""
import argparse
import functools
import logging
import os
import pickle

import json

import numpy as np
# import wandb

from uncertainty.utils import utils
from uncertainty.utils.eval_utils import (
    bootstrap, compatible_bootstrap, auroc, accuracy_at_quantile,
    area_under_thresholded_accuracy)


utils.setup_logger()

result_dict = {}

SMALL_RUN = True
SMALL_SET_SIZE = 250

if SMALL_RUN==True:
    UNC_MEAS = f'uncertainty_measures_max_idx_{SMALL_SET_SIZE}_withreversemap.pkl' #f'uncertainty_measures_max_idx_{SMALL_SET_SIZE}.pkl'
    RESULTS_FILE = f'analysis_result_max_idx_{SMALL_SET_SIZE}_withreversemap.json' # f'analysis_result_max_idx_{SMALL_SET_SIZE}.json'
else:
    UNC_MEAS = 'uncertainty_measures.pkl'
    RESULTS_FILE = 'analysis_result.json'
    
ROOT_DIR = ''#THIS NEEDS TO BE UPDATED 


def analyze_run(
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


if __name__ == '__main__':
    RUN_ID = 'qsy0i0cb'
    analyze_run(wandb_runids=RUN_ID)
