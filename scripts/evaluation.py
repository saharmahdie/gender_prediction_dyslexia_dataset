import os
import sys
import logging
import datetime
import time
import json

sys.path.append('../')

from schau_mir_in_die_augen.evaluation.base_selection import get_dataset, parse_arguments, start_logging
from schau_mir_in_die_augen.evaluation.base_evaluation import FeatureLabels

from schau_mir_in_die_augen.small_functions import load_pickle_file

from scripts.inspection import check_python_version


check_python_version()

def main(args):

    ###############
    # PREPARATION #
    ###############
    folder_attachment = '_{data}_{clf}_{method}'.format(data=args.dataset, clf=args.classifier, method=args.method)
    start_logging('Evaluation', folder_attachment=folder_attachment)
    logger = logging.getLogger(__name__)
    logger.info('Start Evaluation with Parameters:\n{}'.format(args))
    time_start = time.time()
    ###############

    # dataset selection
    ds = get_dataset(args.dataset, args)

    # load pre-trained evaluation
    eval_file_name = args.modelfile
    eva = load_pickle_file(eval_file_name)

    ############
    # LOADING #
    ############
    logger.info('Start Loading (Preparing took {:.3f} seconds).'.format(time.time()-time_start))
    time_loading = time.time()
    start_time = datetime.datetime.now()
    ############

    # todo: this is strange, wy we load train data in test?
    # data for training, we load them to check whether all testing labels are known from training
    trajectories_train = ds.load_training_trajectories()
    logger.debug('{num} Cases for Training: {cas}'.format(num=len(trajectories_train),
                                                          cas=trajectories_train.cases))

    trajectories_train = eva.do_data_preparation(trajectories_train)

    # save the processing steps
    logger.debug('Processing of Training: {}'.format(trajectories_train.processing))

    # data for testing
    if args.test_dataset:
        print('Evaluating on different dataset!')
        test_ds = get_dataset(args.test_dataset, args)
        assert type(ds) == type(test_ds), \
            'Completely different datasets have different IDs and cannot be used in transfer eval'
        trajectories_test = test_ds.load_testing_trajectories()
        if not set(trajectories_test.users).issubset(set(trajectories_train.users)):
            print('Warning: Not all test labels are present in the training labels. Using a common subset.')
            trajectories_test.select_users(list(set(trajectories_test.users) & set(trajectories_train.users)))
    else:
        trajectories_test = ds.load_testing_trajectories()
        test_ds = ds

    logger.debug('{num} Cases for Testing: {cas}'.format(num=len(trajectories_test),
                                                         cas=trajectories_test.cases))
    trajectories_test = eva.do_data_preparation(trajectories_test)

    logger.debug('Processing of Testing: {}'.format(trajectories_test.processing))

    print("data loading time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))

    ############
    # ANALYSIS #
    ############
    extract_time = time.time()
    extract_time2 = datetime.datetime.now()  # todo: remove this

    logger.info('Start Extraction (Loading test and train data took {:.3f} seconds).'.format(
        time.time() - time_loading))

    print("Feature extraction for {} testing cases".format(len(trajectories_test)))
    labeled_feature_values_test = eva.provide_feature(trajectories=trajectories_test,
                                                      normalize=args.use_normalization,
                                                      label=args.label)

    print("Feature extraction time: ", str(datetime.timedelta(seconds=
                                                              (datetime.datetime.now() - extract_time2).seconds)))

    # TODO: move this to another script, we don't have the training features here (cs)


    ##############
    # EVALUATING #
    ##############
    time_evaluating = time.time()
    ##############

    result = []

    # really hacky way to do all the evals. we can make this much smarter!
    sessions_per_subject = len(trajectories_test) // len(test_ds.get_users())
    # 1 to 10 -> step 1
    for s in range(1, 11):
        if s > sessions_per_subject:
            continue
        print(f'Samples per subject: {s}')
        result = eval_subset(labeled_feature_values_test, eva, args, s)
        # will use the model trained on top features for evaluation if top_features_train = True
    # 20 to 60 -> step 10
    for s in range(20, 70, 10):
        if s > sessions_per_subject:
            continue
        print(f'Samples per subject: {s}')
        result = eval_subset(labeled_feature_values_test, eva, args, s)
        # will use the model trained on top features for evaluation if top_features_train = True
    # 100 to max -> step 100
    for s in range(100, sessions_per_subject+100, 100):
        if s > sessions_per_subject:
            continue
        print(f'Samples per subject: {s}')
        result = eval_subset(labeled_feature_values_test, eva, args, s)
        # will use the model trained on top features for evaluation if top_features_train = True

    ############
    # FINISHED #
    ############
    logger.info('Start Training (Loading test and train data took {:.3f} seconds).'.format(time.time()-time_evaluating))
    logger.info('Complete Program took {:.3f} seconds.'.format(time.time() - time_start))
    ############
    print("total time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))

    return result   # only last result (will be overwritten often)


def eval_subset(labeled_feature_values, eva, args, subset_length):
    """ Evaluate subsets with different number of cases inside"""

    feature_values_subset = []

    # get unique and sorted user_names
    user_names = sorted(list(set(labeled_feature_values[0][FeatureLabels.user])))
    for ids in labeled_feature_values[1:]:
        assert user_names == sorted(list(set(ids[FeatureLabels.user])))

    for ii in range(len(labeled_feature_values)):
        feature_values_subset.append([])
        for user_name in user_names:
            labeled_feature_values_user = labeled_feature_values[ii].loc[
                labeled_feature_values[ii][FeatureLabels.user] == user_name]
            # reduced number of cases
            case_names = list(set(labeled_feature_values_user[FeatureLabels.case]))[:subset_length]
            # todo: This is horrible inefficent and dangerous - case_names could be sorted differently ...
            for case_name in case_names:
                feature_values_subset[ii].extend(labeled_feature_values_user.loc[
                                                     labeled_feature_values_user[FeatureLabels.case] == case_name])
    print('W: subset_length is deactivated')

    res = eva.evaluation(labeled_feature_values)
    args.whl_test_samples = subset_length
    res['config'] = vars(args)
    result_name = '&'.join([f'{k}={v}' for k, v in res['config'].items()])
    res['params'] = result_name
    filename = 'results/file_' + str(datetime.datetime.now()).replace(':', '') + '.json'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if 'M' in res: del res['M']
    if 'F' in res: del res['F']
    with open(filename, 'w') as outfile:
        json.dump(res, outfile)

    return res


def get_user_ids(labeled_feature_values):
    """ extract user_ids from labeled_feature_values.

    It is necessary because labeled_feature_values is a list with unknown length.
        Each entry is a DataFrame and could have different user_ids

    todo: It should be used elsewhere too - i know there are some worse code lines for this somewhere"""

    user_ids = set()
    for lfv in labeled_feature_values:
        user_ids = user_ids | set(lfv[FeatureLabels.user])

    return list(user_ids)


if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])

    print(args)

    main(args)

