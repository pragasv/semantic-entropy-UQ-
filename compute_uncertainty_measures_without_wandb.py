"""Compute uncertainty measures after generating answers. - without wandb"""
from collections import defaultdict
import logging
import os
import json
import pickle
import numpy as np
# import wandb
from tqdm import tqdm
from scipy.interpolate import interp1d

from analyze_results import analyze_run
from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures.p_ik import get_p_ik
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from uncertainty.uncertainty_measures import p_true as p_true_utils
from uncertainty.utils import utils

from uq_utils_new.utils_qcm import generate_test_KME, generate_hamiltonian_testing, generate_hamiltonians
from uq_utils_new.utils_UQ import try_rishab_UQ
from statsmodels.nonparametric.bandwidths import bw_silverman as bw

utils.setup_logger()

EXP_DETAILS = 'experiment_details.pkl'
ROOT_DIR = "" #THIS NEEDS TO BE UPDATED 
OBJECT_LOCATION = "analysis"

SMALL_SET = True
SMALL_SET_SIZE = 250

def try_rishab_UQ_over_iterations(perturb_eigen_vectors, sigma_kernel = 0.6):
    L, M, runs = perturb_eigen_vectors.shape

    ratio_main = np.zeros((L,M), dtype=np.complex64)

    for i in range(L):
        input_array = np.abs(perturb_eigen_vectors[i,:, :])
        laplacian_array = np.gradient(np.gradient(input_array, axis=0), axis=0)

        mean_ratio_array = np.mean(((sigma_kernel**2)/2) * laplacian_array/input_array, axis=1)

        ratio_main[i,:] = mean_ratio_array - np.min(mean_ratio_array)
    
    return ratio_main

# Convert all ndarrays to lists before saving
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    else:
        return obj


def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, complex):
        return {'real': obj.real, 'imag': obj.imag}
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def UQ_spinchain(input_signal):
    if np.any(np.isnan(np.std(input_signal))) or np.any(np.isinf(np.std(input_signal))):
        pass
    else:
        input_signal = (input_signal - np.mean(input_signal)) / np.std(input_signal)
        ### input array not normalizable - same values ()
    elements = [1, 2, 3]

    # Define the number of sequences to generate
    silverman_bw = 5
    num_operators = 363
    N_num_data = 128
    max_bits = int(np.log2(N_num_data))
    Hamiltonians_list = generate_hamiltonians(max_bits, elements, num_operators)
    x = np.linspace(-3, 3, N_num_data)

    sigg = silverman_bw * np.average(bw(input_signal))

    reshaped_psi = generate_test_KME(sigma_kernel = sigg, signal_name="input_given", N_num_data = N_num_data, x=x, x_data=input_signal)

    lambda_val = 1
    perturbation_amplitude = 0.001

    try: 
        per_change_out_array, MAE_array, perturbed_energy_array, unperturbed_energy_array, unperturbed_eigenvectors_array_forplot, \
            perturbed_eigenvectors_array_forplot, Hamiltonian_returned, returned_null_space_vector, \
                unperturbed_eigenvectors_array_raw, perturbed_eigenvectors_array_raw, perturbed_eigenvectors_array_all_modes_together, start_idx_all = generate_hamiltonian_testing(reshaped_psi, 
                                                                                                                                Hamiltonians_list, 
                                                                                                                                lambda_value=lambda_val, 
                                                                                                                                return_hamiltonian=True,
                                                                                                                                perturbation_amplitude = perturbation_amplitude,
                                                                                                                                return_raw_values=True
                                                                                                                                )
    except np.linalg.LinAlgError: 
        raise ValueError

    ratio_main_perturbed = try_rishab_UQ(perturbed_eigenvectors_array_forplot.T, sigma_kernel=sigg)
    return ratio_main_perturbed, perturbed_eigenvectors_array_forplot, reshaped_psi, unperturbed_eigenvectors_array_forplot



def UQ_map_todata_algorithm_remap(ratio_main_perturbed, x_input, pr, interpolate=True):
    ''' 
    Map the bands which you get from std of UQ 
    to the model prediction values 
    '''
    UQ_data = np.zeros((pr.shape[0], 8), dtype=np.complex64)
    
    for mode in range(8):
        ### interpolate ratio_main_perturbed
        if interpolate:
            interpolation_function = interp1d(x_input, ratio_main_perturbed[:,mode], kind='linear', fill_value="extrapolate")
            UQ_data[:, mode] = np.abs(interpolation_function(pr))
        else:
            UQ_data[:, mode] = ratio_main_perturbed[:,mode].reshape(-1,1)
    return UQ_data.real


def predictive_entropy_renyi(log_probs, order=2):
    entropy = - np.log(np.sum(np.power(np.exp(log_probs), order)))
    return entropy 

def save_file(object, RUN_ID, file):
    ## create folder - if not exist
    os.makedirs(f'{ROOT_DIR}/{RUN_ID}/files/{OBJECT_LOCATION}', exist_ok=True)
    
    with open(f'{ROOT_DIR}/{RUN_ID}/files/{OBJECT_LOCATION}/{file}', 'wb') as f:
        pickle.dump(object, f)

def save_file_json(object, RUN_ID, file):
    ## create folder - if not exist
    os.makedirs(f'{ROOT_DIR}/{RUN_ID}/files/{OBJECT_LOCATION}', exist_ok=True)
    
    with open(f'{ROOT_DIR}/{RUN_ID}/files/{OBJECT_LOCATION}/{file}', 'w') as f:
        json.dump(object, f, indent=4)


def main(args, RUN_ID='run-20250311_192042-qsy0i0cb'):
    count_issues = 0

    user = os.environ['USER']
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"


    def restore(filename):
        class Restored:
            name = f'{ROOT_DIR}/{RUN_ID}/files/{filename}'

        return Restored

    train_generations_pickle = restore('train_generations.pkl')
    with open(train_generations_pickle.name, 'rb') as infile:
        train_generations = pickle.load(infile)

    # Load entailment model.
    if args.compute_predictive_entropy:
        logging.info('Beginning loading for entailment model.')
        if args.entailment_model == 'deberta':
            entailment_model = EntailmentDeberta()
        elif args.entailment_model == 'gpt-4':
            entailment_model = EntailmentGPT4(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-3.5':
            entailment_model = EntailmentGPT35(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-4-turbo':
            entailment_model = EntailmentGPT4Turbo(args.entailment_cache_id, args.entailment_cache_only)
        elif 'llama' in args.entailment_model.lower():
            entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
        else:
            raise ValueError
        logging.info('Entailment model loading complete.')

    if args.compute_p_true_in_compute_stage:
        # This is usually not called.
        old_exp = restore(EXP_DETAILS)
        with open(old_exp.name, "rb") as infile:
            old_exp = pickle.load(infile)

        if args.reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            pt_model = utils.init_model(old_exp['args'])

        pt_train_dataset, pt_validation_dataset = load_ds(
            old_exp['args'].dataset, add_options=old_exp['args'].use_mc_options,
            seed=args.random_seed)
        del pt_validation_dataset

        # Reduce num generations used in p_true if needed!
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            num_gen = args.use_num_generations
        else:
            num_gen = args.num_generations

        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=pt_model,
            dataset=pt_train_dataset,
            indices=old_exp['p_true_indices'],
            prompt=old_exp['prompt'],
            brief=old_exp['BRIEF'],
            brief_always=old_exp['args'].brief_always and old_exp['args'].enable_brief,
            make_prompt=utils.get_make_prompt(old_exp['args']),
            num_generations=num_gen,
            metric=utils.get_metric(old_exp['args'].metric))
        del p_true_responses

        logging.info('Generated few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    if args.recompute_accuracy:
        # This is usually not enabled.
        logging.warning('Recompute accuracy enabled. This does not apply to precomputed p_true!')
        metric = utils.get_metric(args.metric)

    # Restore outputs from `generate_answrs.py` run.
    result_dict_pickle = restore('uncertainty_measures.pkl')
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)
    result_dict['semantic_ids'] = []

    validation_generations_pickle = restore('validation_generations.pkl')
    with open(validation_generations_pickle.name, 'rb') as infile:
        validation_generations = pickle.load(infile)

    entropies = defaultdict(list)
    cluster_details = defaultdict(lambda: {'semantic_idcs': [], 
                                           'responses': [], 
                                           'spinchain_UQ_KME':[], 
                                           'spinchain_UQ_perturbed_eigenvectors':[], 
                                           'spinchain_UQ_KME_UQ':[],
                                           'spinchain_UQ_KME_UQ_std':[],
                                           'semantic_renyi_entropy':[],
                                           'semantic_entropy':[],
                                           'regular_entropy':[],
                                           'log_liks_agg': [],
                                           'KME_UQ_reversemapped': [],
                                           'KME_UQ_std_reversemapped': []
                                        })
    UQ_spin_attributes = {}
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    p_trues = []
    count = 0  # pylint: disable=invalid-name

    def is_answerable(generation):
        return len(generation['reference']['answers']['text']) > 0
    
    if (SMALL_SET== True):
        uncertain_measure_file_name = f'uncertainty_measures_max_idx_{SMALL_SET_SIZE}_withreversemap.pkl'
        spinchain_measure_file_name = f'spinchain_measures_max_idx_{SMALL_SET_SIZE}_withreversemap.json'
    else:
        uncertain_measure_file_name = 'uncertainty_measures_withreversemap.pkl'
        spinchain_measure_file_name = 'spinchain_measures_withreversemap.json'

    # Loop over datapoints and compute validation embeddings and entropies.
    for idx, tid in tqdm(enumerate(validation_generations)):
        if idx == 10:
            pass
        
        if (idx >= SMALL_SET_SIZE) and (SMALL_SET== True):
            break

        example = validation_generations[tid]
        question = example['question']
        context = example['context']
        full_responses = example["responses"]
        most_likely_answer = example['most_likely_answer']

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        if args.recompute_accuracy:
            logging.info('Recomputing accuracy!')
            if is_answerable(example):
                acc = metric(most_likely_answer['response'], example, None)
            else:
                acc = 0.0  # pylint: disable=invalid-name
            validation_is_true.append(acc)
            logging.info('Recomputed accuracy!')

        else:
            validation_is_true.append(most_likely_answer['accuracy'])

        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer['embedding'])
        logging.info('validation_is_true: %f', validation_is_true[-1])

        if args.compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[:args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            for i in log_liks:
                assert i

            if args.compute_context_entails_response:
                # Compute context entails answer baseline.
                entropies['context_entails_response'].append(context_entails_response(
                    context, responses, entailment_model))

            if args.condition_on_question and args.entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            # Compute semantic ids.
            semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=args.strict_entailment, example=example)

            result_dict['semantic_ids'].append(semantic_ids)
            
            # Compute entropy from frequencies of cluster assignments.
            entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))

            # Length normalization of generation probabilities.
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]


            ### spinchain UQ
            try:
                ratio_main_perturbed, perturbed_eigenvectors_array_save, KME, eigen_vectors = UQ_spinchain(log_liks_agg)
                KME_UQ = np.mean(ratio_main_perturbed, axis=(0,1)).real
                KME_UQ_std = np.std(ratio_main_perturbed, axis=(0,1)).real

                N_num_data = 128
                x_input = np.linspace(-3, 3, N_num_data)
                UQ_remapped = UQ_map_todata_algorithm_remap(ratio_main_perturbed, x_input, np.array(log_liks_agg))
                KME_UQ_reversemapped = np.mean(UQ_remapped, axis=1)
                KME_UQ_std_reversemapped = np.std(UQ_remapped, axis=1)

                UQ_spin_attributes[tid] = {'ratio_main_perturbed': ratio_main_perturbed,
                                        'perturbed_eigenvectors': perturbed_eigenvectors_array_save,
                                        'KME': np.float64(KME),
                                        'eigen_vectors': eigen_vectors,
                                        'KME_UQ': np.float64(KME_UQ),
                                        'KME_UQ_std': np.float64(KME_UQ_std)
                                            } 
            except ValueError:
                KME_UQ = 0.0
                count_issues += 1
                pass

            # Compute naive entropy.
            regular_entropy_used = predictive_entropy(log_liks_agg)
            entropies['regular_entropy'].append(regular_entropy_used)

            # Compute semantic entropy.
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies['semantic_entropy'].append(pe)

            ### CHANGE(praga)compute: 2nd order Renyis 
            re = predictive_entropy_renyi(log_likelihood_per_semantic_id)
            entropies['semantic_renyi_entropy'].append(re)

            entropies['KME_UQ'].append(np.float64(KME_UQ))
            entropies['KME_UQ_std'].append(np.float64(KME_UQ_std))

            entropies['KME_UQ_reversemapped'].append(np.float64(np.mean(UQ_remapped, axis=(0,1))))
            entropies['KME_UQ_std_reversemapped'].append(np.float64(np.std(UQ_remapped, axis=(0,1))))


            ## extract cluster details             
            cluster_details[tid]['semantic_idcs'].append(semantic_ids)
            cluster_details[tid]['responses'].append(responses)
            cluster_details[tid]['spinchain_UQ_KME'].append(np.float64(KME))   
            cluster_details[tid]['spinchain_UQ_perturbed_eigenvectors'].append(perturbed_eigenvectors_array_save)   
            cluster_details[tid]['spinchain_UQ_KME_UQ'].append(np.float64(KME_UQ))   
            cluster_details[tid]['spinchain_UQ_KME_UQ_std'].append(np.float64(KME_UQ_std))
            cluster_details[tid]['semantic_renyi_entropy'].append(re)
            cluster_details[tid]['semantic_entropy'].append(pe)
            cluster_details[tid]['regular_entropy'].append(regular_entropy_used)
            cluster_details[tid]['log_liks_agg'].append(log_liks_agg)
            cluster_details[tid]['KME_UQ_reversemapped'].append(np.float64(KME_UQ_reversemapped))
            cluster_details[tid]['KME_UQ_std_reversemapped'].append(np.float64(KME_UQ_std_reversemapped))


            # pylint: disable=invalid-name
            log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])
            # pylint: enable=invalid-name
            logging.info(80*'#')
            logging.info('NEW ITEM %d at id=`%s`.', idx, tid)
            logging.info('Context:')
            logging.info(example['context'])
            logging.info('Question:')
            logging.info(question)
            logging.info('True Answers:')
            logging.info(example['reference'])
            logging.info('Low Temperature Generation:')
            logging.info(most_likely_answer['response'])
            logging.info('Low Temperature Generation Accuracy:')
            logging.info(most_likely_answer['accuracy'])
            logging.info('High Temp Generation:')
            logging.info([r[0] for r in full_responses])
            logging.info('High Temp Generation:')
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        if args.compute_p_true_in_compute_stage:
            p_true = p_true_utils.calculate_p_true(
                pt_model, question, most_likely_answer['response'],
                responses, p_true_few_shot_prompt,
                hint=old_exp['args'].p_true_hint)
            p_trues.append(p_true)
            logging.info('p_true: %s', np.exp(p_true))

        count += 1
        if count >= args.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break
    
    logging.info('KME issue counts: %d', count_issues)
    logging.info('Accuracy on original task: %f', np.mean(validation_is_true))
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    logging.info('Unanswerable prop on validation: %f', np.mean(validation_unanswerable))

    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)

    if args.compute_p_ik or args.compute_p_ik_answerable:
        # Assemble training data for embedding classification.
        train_is_true, train_embeddings, train_answerable = [], [], []
        for tid in train_generations:
            most_likely_answer = train_generations[tid]['most_likely_answer']
            train_embeddings.append(most_likely_answer['embedding'])
            train_is_true.append(most_likely_answer['accuracy'])
            train_answerable.append(is_answerable(train_generations[tid]))
        train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
        train_unanswerable = [0.0 if is_t else 1.0 for is_t in train_answerable]
        logging.info('Unanswerable prop on p_ik training: %f', np.mean(train_unanswerable))

    if args.compute_p_ik:
        logging.info('Starting training p_ik on train embeddings.')
        # Train classifier of correct/incorrect from embeddings.
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_is_false,
            eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
        result_dict['uncertainty_measures']['p_ik'] = p_ik_predictions
        logging.info('Finished training p_ik on train embeddings.')

    if args.compute_p_ik_answerable:
        # Train classifier of answerable/unanswerable.
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_unanswerable,
            eval_embeddings=validation_embeddings, eval_is_false=validation_unanswerable)
        result_dict['uncertainty_measures']['p_ik_unanswerable'] = p_ik_predictions

    if args.compute_p_true_in_compute_stage:
        result_dict['uncertainty_measures']['p_false'] = [1 - p for p in p_trues]
        result_dict['uncertainty_measures']['p_false_fixed'] = [1 - np.exp(p) for p in p_trues]

    # utils.save(result_dict, 'uncertainty_measures.pkl')
    save_file(result_dict, RUN_ID, uncertain_measure_file_name)

    serializable_data = make_json_serializable(dict(cluster_details))
    save_file(serializable_data, RUN_ID, spinchain_measure_file_name)

    if args.compute_predictive_entropy:
        entailment_model.save_prediction_cache()
