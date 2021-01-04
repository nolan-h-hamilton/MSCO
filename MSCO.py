"""
MSCO provides tools for constructing and optimizing multi-stage classifiers
from ordered feature set partitions
"""

import random
import math
import numpy as np
import pandas as pd
import jenkspy as jenks
from sklearn.base import clone
from operator import itemgetter
from sklearn.utils import shuffle
from sklearn.preprocessing import minmax_scale


def check_partition_criteria(partition, phase_feature_mins = None,
                             phase_feature_reqs = None):

    """check if `partition` satisfies given criteria

    Args:
        partition: a list containing integer elements with the value at
          index i denoting the stage assignment of feature i
        phase_feature_mins: a list containing integer elements with
          the value at index i denoting the minimum number of features
          that must be assigned to stage i.
        phase_feature_reqs: a list containing integer elements with
          the value at index i denoting the required stage assignment
          for feature i. if no required stage assignment is desired 
          for feature i, then phase_feature_reqs[i] should be set to
          -1.

    Returns:
        True if criteria is satisfied
    """

    #first check that there are no empty stages
    for i in range(0,max(partition) - 1):
        if i not in partition:
            return False

    if phase_feature_reqs is not None:
        for i, req in enumerate(phase_feature_reqs):
            if req != -1 and req != partition[i]:
                return False

    if phase_feature_mins is not None:
        for i, min_ in enumerate(phase_feature_mins):
            if partition.count(i) < min_:
                return False

    return True


def partition_dataset(X, partition):

    """partition dataframe `X` according to `partition`

    Args:
        X: a pandas dataframe containing records for training/testing
        partition: a list containing integer elements with the value at
          index i denoting the stage assignment of feature i

    Returns:
        a list of pandas dataframes representing the partitioned dataset
    """

    last_stage_index = max(partition) + 1
    partitioned_X = []
    k_th = []
    for k in range(0,last_stage_index):
        k_th = [i for i, val in enumerate(partition) if val <= k ]
        partitioned_X.append(X.iloc[:,k_th])
    return partitioned_X


def randomize_partition(partition, n, max_stage_ct,
                        phase_feature_reqs=[], phase_feature_mins=[],
                        p_add=0.05, p_mod=0.25):

    """generate random population of partitions from `partition`

    Args:
        partition: a list containing integer elements with the value at
          index i denoting the stage assignment of feature i
        n: number of new partitions to generate
        p_add: probability of adding a new stage
        p_mod: probability of modifying the existing stage assignment
        max_stage_ct: stage assignment values will not be set higher than this

    Raises:
        ValueError: if `partition` does not satisfy criteria

    Returns:
        a list of lists representing partitions of the initial feature set.
    """

    # check if partition is valid
    if not check_partition_criteria(
      partition, phase_feature_mins, phase_feature_reqs):
        raise ValueError(f'supplied partition `partition`: {partition} invalid')

    current_stage_max = max(partition)
    random_pop = []
    for i in range(n):
        random_part = partition
        for j in range(len(partition)):
            # ensure changing stage assignment will not create emtpy stage
            if random_part.count(random_part[j]) == 1:
                continue
            
            # I: add new stage if allowed
            if random.random() < p_add and partition[j] < max_stage_ct-1:
                    random_part[j]  += 1
                    if partition[j] > current_stage_max:
                        current_stage_max = partition[j]
                    continue

            # II: modify stage assignment of feature j
            elif random.random() < p_mod:
                if current_stage_max > 1:
                    random_part[j] = random.randrange(0, current_stage_max)
                else:
                    random_part[j] = random.randrange(0, 2)

        # finally, add the generated partition
        random_pop.append(random_part)

    return random_pop


def make_stages(clf, X, y, partition):
    
    """create a multi-stage classification scheme according to `partition`

    Args:
        clf: sklearn classifier object
`       X: pandas dataframe containing training and test records
        y: array-like object containing labels for corresponding records
        partition: array-like object containing stage assignments for 
            features. partition[i] is the stage assignment for feature
            i.

    Returns:
        partitioned dataframe AND a list of stages/subclassifiers
    """
    
    #partition dataset and train submodels for each stage
    parted_X = partition_dataset(X, partition)
    trained_models = []
    for df in parted_X:
        trained_models.append(clone(clf).fit(df, y))
    return parted_X, trained_models


def process_input(parted_X, trained_models, x, prob_thresh=.75):
    
    """evaluate inputs with multi-stage model

    Args:
        parted_X: returned from make_stages()[0], this is a subset
          of X containing only features included in stage i
        trained_models: submodels (stages) trained on parted_X
        x: the input to be processed. must have all features in
          initial dataset (X)
        prob_thresh: probability threshold for conclusiveness

    Returns:
        an index associated with the respective label/class. 
        if processing is not conclusive, -1 is returned
    """
    
    x = pd.Series(x)
    for i in range(0, len(parted_X)):
        test_record = x[parted_X[i].columns]
        class_probabilites = trained_models[i].predict_proba(x)
        mcp = np.amax(class_probabilities)
        
        if mcp >= prob_thresh:
            return np.where(class_probabilities == mcp)
    return -1


def staged_classify(clf, X, y, partition, feature_costs, 
                    prob_thresh=.75,
                    train_percent=.8,
                    all_costs=[]):

    """evaluate the accuracy/efficiency of a multi-stage model given by `partition`

    Args:
        clf: scikit-learn classifier object
        X: pandas df of records for training/testing
        y: labels for records in `X`
        partition: a list containing integer elements with the value at
          index i denoting the stage assignment of feature i
        feature_costs: list of values corresponding to the runtime costs
          of features in `X`. 
        prob_thresh: the minimum confidence in prediction that a stage
          in the model must have for a decision to be deemed conclusive
        train_percent: this value denotes the percentage of records that 
          will be dedicated to training. (1 - `train-percent`) records 
          will be used for testing

    Returns:
        a dictionary related to the performance of the model:
          'correct': number of correctly classified records
          'incorrect': number of incorrectly classified records
          'inconclusive': number of records determined 'inconclusive'
          'runtime': total runtime of model according to values in `feature_costs`
    """
    
    # shuffle data, split into train/test
    X, y = shuffle(X, y)
    train_index = math.floor(len(X)*train_percent)
    train_X, train_y = X.iloc[:train_index], y[:train_index]
    test_X, test_y = X.iloc[train_index:], y[train_index:]
    test_X.reset_index(drop=True, inplace=True)
    
    #partition dataset and train submodels for each stage
    parted_X = partition_dataset(train_X, partition)
    trained_models = []
    for df in parted_X:
        trained_models.append(clone(clf).fit(df, train_y))

    # run through test records using multi-stage model
    # compute performance factors incrementally
    correct = 0.0
    incorrect = 0.0
    inconclusive = 0.0
    runtime = 0.0
    for j, row in test_X.iterrows():
        is_conclusive = False
        used_features = []
        i = 0
        while not is_conclusive and i < len(trained_models):
            # get stage i features and classify            
            test_record = row[parted_X[i].columns]
            pred_conf = trained_models[i].predict_proba([np.array(test_record)]).flat
            max_conf = max(pred_conf)
            pred_label = np.where(pred_conf==max_conf)[0][0]


            # if conclusive at stage i, check if correct

            if max_conf >= prob_thresh:
                if test_y[j] == pred_label:
                    correct += 1.0
                else:
                    incorrect += 1.0
                is_conclusive = True

            # runtime is the sum of feature costs used for each record
            for col in np.nditer(parted_X[i].columns):
                # do not double-count feature costs
                if col not in used_features:
                    try:
                        runtime += feature_costs[col]

                    except IndexError:
                        runtime += all_costs[col]
                    used_features.append(col)
            i += 1
        # after loop is executed, check if still inconclusive
        if not is_conclusive:
            inconclusive += 1.0

    return {'correct': correct, 'incorrect': incorrect,
            'inconclusive': inconclusive, 'runtime': runtime,
            'num_test_records': (len(X) - train_index)}


def jenks_stages(clf, X, y, feature_costs, max_k,
                 min_max_norm=True, prob_thresh=.9,
                 train_percent=.75):
    
    """use jenks natural breaks optiimization on costs to create multi-stage models

    Args:
        clf: sklearn classifier object
        X: pandas dataframe containing records for training/testing
        y: array-like object containing labels for corresponding records
        feature_costs: array-like object containing float values corresponding to
            "costs" of features as defined in MSCO/docs/hamilton_thesis.pdf
        max_k: maximum number of allowed stages to consider
        min_max_norm: if True, scale feature costs to [0,1]

    Returns:
        a list of solutions generated from jenks using varying K in [2, max_k]
            along with their respective performance
    """

    solutions = []

    # values are sorted by jenks() by default
    # sorted_features=sorted(feature_costs)
    sorted_features = feature_costs
    
    if min_max_norm:
        sorted_features = minmax_scale(sorted_features)
        feature_costs = minmax_scale(feature_costs)
        
    # determine breaks in feature costs    
    breaks = [jenks.jenks_breaks(sorted_features, i+1)
                 for i in range(1,max_k)]

    for brk in breaks:
        # bin features according to breaks
        partition = [x for x in np.digitize(feature_costs, brk ,True)]
        
        # evaluate arrangement given by breaks
        score = pm_euclidean(staged_classify(clf, X, y, partition,
                                             feature_costs,
                                             prob_thresh,
                                             train_percent))
        solutions.append([partition, score])

    return solutions


def n_stages(clf, X, y, feature_costs,
                   min_max_norm=True,
                   prob_thresh=.9,
                   train_percent=.75):

    """create an N-stage model of increasing feature cost

    Args:
        clf: sklearn classifier object
        X: pandas dataframe containing records for training/testing
        y: array-like object containing labels for corresponding records
        feature_costs: array-like object containing float values corresponding to
            "costs" of features as defined in MSCO/docs/hamilton_thesis.pdf

    Returns:
        n-stage solution and correponding performance score
    """
    
    partition = []

    if min_max_norm:
        feature_costs = minmax_scale(feature_costs)
        
    # create mapping from unsorted to sorted costs
    sorted_indices = list(np.argsort(feature_costs))
    
    for i in range(len(feature_costs)):
        # assn. for a feature is index of feature
        # in sorted mapping from feature costs
        assignment = sorted_indices.index(i)
        partition.append(assignment)

    score = pm_euclidean(staged_classify(clf, X, y, partition,
                                         feature_costs,
                                         prob_thresh,
                                         train_percent))
    return score


def beam(clf, X, y, partition, feature_costs, pop_size=50, max_iter=10,
         beam_percent=0.1, stage_inc_max=3):

    """perform a simple beam search on the solution space of multi-stage designs

    Args:
        clf: scikit-learn classifier object
        X: pandas df of records for training/testing
        y: labels for records in `X`
        partition: a list containing integer elements with the value at
          index i denoting the stage assignment of feature i
        feature_costs: list of values corresponding to the runtime costs
          of features in `X`. 
        pop_size: denotes the size of generated population from which to
          select beam population
        max_iter: maximum number of generations (populations)

    Returns:
        a list containing the best performing partition and its performance
    """
    
    part_cpy = partition
    
    # will be referenced multiple times
    beam_pop_size = math.floor(pop_size*beam_percent) + 1

    # create initial population
    initial_pop = randomize_partition(partition, pop_size,
      max_stage_ct = max(partition) + stage_inc_max)

    # evaluate and rank initial population
    ranked_pop = []
    for member in initial_pop:
        member_scores = staged_classify(clf, X, y, member, feature_costs)
        performance = pm_euclidean(member_scores)
        ranked_pop.append([member, performance])
    ranked_pop.sort(key=lambda x: x[1], reverse=True)

    # carry out selection of beam population and modify over `max_iter` gens
    for i in range(max_iter):
        best_part = ranked_pop[0][0]
        best_perf = ranked_pop[0][1]
        print(f'gen: {i}, best_partition: {best_part}, performance: {best_perf}')
        
        # take beam_pop_size members of previous pop and form beam population
        beam_pop = ranked_pop[:beam_pop_size]

        new_pop = beam_pop
        while len(new_pop) < pop_size:
            # generate new member of population by randomizing beam pop
            new_mem = randomize_partition(
              beam_pop[random.randrange(0, beam_pop_size)][0],
              1, max(part_cpy) + stage_inc_max)

            # score and add new member
            new_mem_scores = staged_classify(clf, X, y, new_mem[0], feature_costs)
            new_pop.append([new_mem[0], pm_euclidean(new_mem_scores)])

        # now that new population is generated and scored, sort for beam
        new_pop.sort(key=lambda x: x[1], reverse=True)
        ranked_pop = new_pop
    print('final gen: {}'.format(ranked_pop[0]))
    return ranked_pop[0]


def deterministic_assn(clf, X, y, K, seed, feature_costs, max_iter=3):

    """deterministic sequential feature assignment method described in 3.6.1

    Args:
        clf: sklearn classifier object
        X: pandas dataframe containing records for training/testing
        y: array-like object containing labels for corresponding records
        feature_costs: array-like object containing float values corresponding to
            "costs" of features as defined in MSCO/docs/hamilton_thesis.pdf
        seed: beginning partition to modify iteratively with deterministic method
        max_iter: number of iterations through all feature assignments, modifying
            each feature assignment on each iteration

    Returns:
        solution obtained after `max_iter` runs through list of feature assignments.
"""
        
    final_partition = seed
    sols = []
    for h in range(max_iter):
        
        print("iter: {}\n".format(h))
        
        for i, val in enumerate(final_partition):
            
            print(final_partition)
            max_score = 0
            max_assn = 0
            
            for j in range(K):
                
                final_partition[i] = j
                score = pm_euclidean(staged_classify(clf,X, y,
                                                     final_partition, feature_costs))
                sols.append([final_partition, score])
                print("{}: {}".format(j, score))
            
                if score > max_score:
                    max_score = score
                    max_assn = j
            
            final_partition[i] = max_assn
        

    return max(sols, key=itemgetter(1))


    
def stochastic_assn(N, K, feature_costs, feature_benefits,
                    init_choices=[]):

    """stochastic sequential feature assignment method described in 3.6.2

    Args:
        clf: sklearn classifier object
        X: pandas dataframe containing records for training/testing
        y: array-like object containing labels for corresponding records
        feature_costs: array-like object containing float values corresponding to
            "costs" of features as defined in MSCO/docs/hamilton_thesis.pdf
        init_choices: array-like object of length K containing initial features
            to assign to stages 0...K-1

    Returns:
        solution and performance obtained 
"""
    
    # least-squares generated surface for computing stage weights
    def L(x,y):
         return (0.06*(x**2) - 1.32*(y**2) + 1.33*(x*y) - 2.00*x + 0.58*y + 1.30)

    # normalize feature costs to [0,1]
    s = sum(feature_costs)
    feature_costs = [float(x)/s for x in feature_costs]
    
    if init_choices == []:
        # pick initial K random features
        init_choices = random.sample(range(N), k=K)
    
    # initialize assignments to -1
    final_partition = [-1 for i in range(N)]

    init_costs = []    
    for i, ic in enumerate(init_choices):
        final_partition[ic] = i

    cost_sums = [0 for i in range(K)]
    benefit_sums = [0 for i in range(K)]
    
    for i, val in enumerate(final_partition):
        if val != -1:
            continue
        weights = [L(cost_sums[j], benefit_sums[j]) for j in range(K)]
        assignment = random.choices(range(K), weights)[0]
        final_partition[i] = assignment
        cost_sums[assignment] += feature_costs[i]
        benefit_sums[assignment] += feature_benefits[i]
    
    return final_partition


def pm_euclidean(scores, acc_coef=1, conc_coef=1, inv_time_coef=1):

    """default performance metric based on euclidean distance from origin
    
    Args:
        scores: scores dict returned from staged_classify()
        acc_coef: accuracy multiplier
        conc_coef: conclusiveness multiplier
        inv_time_coef: inverse time multiplier

    Returns:
        3-dimensional euclidean distance from origin (0,0,0) to
        (scaled_acc, scaled_conc, scaled_inv_time)
    """
    
    scaled_acc = acc_coef * (scores['correct']/scores['num_test_records'])
    scaled_conc = conc_coef * ((scores['correct'] + scores['incorrect']) 
                                / scores['num_test_records'])
    scaled_inv_time = inv_time_coef * (1 + (1/float(scores['runtime'] + 1)))
    return math.sqrt(scaled_acc ** 2
                     + scaled_conc ** 2
                     + scaled_inv_time ** 2)
                               
