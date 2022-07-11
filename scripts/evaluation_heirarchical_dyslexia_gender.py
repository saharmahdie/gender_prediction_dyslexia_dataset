## example:
##  python3 evaluation_heirarchical_dyslexia_gender.py --method gender-selection --ivt_threshold 50 --classifier rbfn --seed 10 --training_data_ratio 0.8 --user_specific_numbers 19 19 19 19 --use_eye_config BOTH_AVG_FEAT --H_D_train_user_num  7 7 7 7
import os
import sys
import logging
import datetime
import json
import time
import pandas as pd
import random
import copy
import math
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from sklearn.preprocessing import StandardScaler
sys.path.append('../')

from schau_mir_in_die_augen.evaluation.base_selection import \
    get_method, get_classifier, get_default_parser, get_conditional_parser, start_logging # to have the features method, clf and the arguments
from schau_mir_in_die_augen.datasets.Dyslexia import Dyslexia   # to use specific dataset
from schau_mir_in_die_augen.evaluation.base_evaluation import FeatureLabels  # to return labels

def compute_balanced_accuracy(y_true, y_pred):
    ## balanced accuracy = ((correct_set1/all set1) + (correct_set2/all set2) + (correct_set3/all set3) + ....)balanced_accuracy_score / N of the sets
    accuracy_balanced = balanced_accuracy_score(y_true, y_pred)
    print("Final Balanced Accuracy: ", accuracy_balanced)

    return accuracy_balanced

def compute_accuracy(y_true, y_pred):
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    correct_prediction = [x == y for (x, y) in zip(y_pred, y_true)]
    print("correct pred: ", correct_prediction)
    accuracy = sum(correct_prediction)/len(correct_prediction)  # # len(correct_prediction) is the lenght of the all
    print("Final Accuracy: ", accuracy, "num of the correct prediction:", sum(correct_prediction), "num of all the testing users:", len(correct_prediction))

    return accuracy

def voting_based_classifier(res, res_healthy, res_dyslexic):
    """ Calculate Voting accuracy.
     Input:
     :param res: results of the first classifier to predict dyslexia
     :param res_healthy: results of the second classifier to predict healthy gender
     :param res_dyslexic: results of the third classifier to predict dyslexic gender

     Returns: accuracy, accuracy_balanced, accuracy_healthy_only, accuracy_dyslexic_only
     """
    y_pred = []
    y_true = res_dyslexic["True_Lable"] # same true lable for test in clf1, clf3 and clf3
    y_true_dyslexic_only = []
    y_pred_dyslexic_only = []
    y_true_healthy_only = []
    y_pred_healthy_only = []

    for i in range(len(res["True_Lable"])):
        # if res["True_Lable"][i] == "dyslexic": ## use True_Lable of CLF1 or deactivate CLF 1
        if res["Predicted_Lable"][i] == "dyslexic":  ## use Predicted_Lable of CLF1 activate CLF 1
            if res_dyslexic["True_Lable"][i] == "M":
                y_true_dyslexic_only.append("M")
            else:
                y_true_dyslexic_only.append("F")

            if res_dyslexic["Predicted_Lable"][i] == "M":
                y_pred.append("M")
                y_pred_dyslexic_only.append("M")
            else:
                y_pred.append("F")
                y_pred_dyslexic_only.append("F")

        else:
            if res_healthy["True_Lable"][i] == "M":
                y_true_healthy_only.append("M")
            else:
                y_true_healthy_only.append("F")

            if res_healthy["Predicted_Lable"][i] == "M":
                y_pred.append("M")
                y_pred_healthy_only.append("M")
            else:
                y_pred.append("F")
                y_pred_healthy_only.append("F")

    print("All subjects")
    accuracy = compute_accuracy(y_true, y_pred)
    print("Healthy only")
    if y_true_healthy_only!=[] and y_pred_healthy_only!=[]:
        accuracy_healthy_only = compute_accuracy(y_true_healthy_only, y_pred_healthy_only)
    else:
        print("No healthy users found. accuracy_healthy_only will be assigned nan.")
        accuracy_healthy_only = math.nan

    print("Dyslexic only")
    if y_true_dyslexic_only!=[] and y_pred_dyslexic_only!=[]:
        accuracy_dyslexic_only = compute_accuracy(y_true_dyslexic_only, y_pred_dyslexic_only)
    else:
        print("No dyslexic users found. accuracy_dyslexic_only will be assigned nan.")
        accuracy_dyslexic_only = math.nan

    print("All subjects balanced")
    accuracy_balanced = compute_balanced_accuracy(y_true, y_pred)

    print("Healthy only balanced")
    if y_true_healthy_only!=[] and y_pred_healthy_only!=[]:
        accuracy_healthy_only_balanced = compute_balanced_accuracy(y_true_healthy_only, y_pred_healthy_only)
    else:
        print("No healthy users found. accuracy_healthy_only_balanced will be assigned nan.")
        accuracy_healthy_only_balanced = math.nan

    print("Dyslexic only balanced")
    if y_true_dyslexic_only != [] and y_pred_dyslexic_only != []:
        accuracy_dyslexic_only_balanced = compute_balanced_accuracy(y_true_dyslexic_only, y_pred_dyslexic_only)
    else:
        print("No dyslexic users found. accuracy_dyslexic_only_balanced will be assigned nan.")
        accuracy_dyslexic_only_balanced = math.nan

    return accuracy, accuracy_balanced, accuracy_healthy_only, accuracy_healthy_only_balanced, accuracy_dyslexic_only, accuracy_dyslexic_only_balanced


def weighting_based_classifier(res, res_healthy, res_dyslexic):
    """ Calculate weighting accuracy.
     Input:
     :param res: results of the first classifier to predict dyslexia
     :param res_healthy: results of the second classifier to predict healthy gender
     :param res_dyslexic: results of the third classifier to predict dyslexic gender

     Returns: Accuracy
     """
    y_pred = []
    y_true_dyslexic_only = []
    y_pred_dyslexic_only = []
    y_true_healthy_only = []
    y_pred_healthy_only = []
    y_true = res_dyslexic["True_Lable"]  # same true lable for test in clf1, clf3 and clf3
    for i in range(len(res["True_Lable"])):
        prob_male = res['dyslexic'][i]*res_dyslexic['M'][i] + res['healthy'][i]*res_healthy['M'][i]
        prob_female = res['dyslexic'][i] * res_dyslexic['F'][i] + res['healthy'][i] * res_healthy['F'][i]
        if prob_female >= prob_male:
            y_pred.append("F")
        else:
            y_pred.append("M")

        if res["Predicted_Lable"][i] == "healthy":
            prob_male_healthy_only = res['healthy'][i]*res_healthy['M'][i]
            prob_female_healthy_only = res['healthy'][i] * res_healthy['F'][i]
            if prob_female_healthy_only >= prob_male_healthy_only:
                y_pred_healthy_only.append("F")
            else:
                y_pred_healthy_only.append("M")

            if res_healthy["True_Lable"][i] == "M":
                y_true_healthy_only.append("M")
            else:
                y_true_healthy_only.append("F")

        elif res["Predicted_Lable"][i] == "dyslexic":
            prob_male_dyslexic_only = res['dyslexic'][i]*res_dyslexic['M'][i]
            prob_female_dyslexic_only = res['dyslexic'][i] * res_dyslexic['F'][i]
            if prob_female_dyslexic_only >= prob_male_dyslexic_only:
                y_pred_dyslexic_only.append("F")
            else:
                y_pred_dyslexic_only.append("M")

            if res_dyslexic["True_Lable"][i] == "M":
                y_true_dyslexic_only.append("M")
            else:
                y_true_dyslexic_only.append("F")

    print("Weighted accuracy of All subjects")
    Weighted_accuracy = compute_accuracy(y_true, y_pred)

    print("Weighted accuracy of Healthy only")
    if y_true_healthy_only!=[] and y_pred_healthy_only!=[]:
        Weighted_accuracy_healthy_only = compute_accuracy(y_true_healthy_only, y_pred_healthy_only)
    else:
        print("No healthy users found. Weighted_accuracy_healthy_only will be assigned nan.")
        Weighted_accuracy_healthy_only = math.nan

    print("Weighted accuracy of Dyslexic only")
    if y_true_dyslexic_only!=[] and y_pred_dyslexic_only!=[]:
        Weighted_accuracy_dyslexic_only = compute_accuracy(y_true_dyslexic_only, y_pred_dyslexic_only)
    else:
        print("No dyslexic users found. Weighted_accuracy_dyslexic_only will be assigned nan.")
        Weighted_accuracy_dyslexic_only = math.nan

    print("Weighted accuracy balanced of All subjects")
    Weighted_accuracy_balanced = compute_balanced_accuracy(y_true, y_pred)

    print("Weighted accuracy balanced Healthy only")
    if y_true_healthy_only!=[] and y_pred_healthy_only!=[]:
        Weighted_accuracy_healthy_only_balanced = compute_balanced_accuracy(y_true_healthy_only, y_pred_healthy_only)
    else:
        print("No healthy users found. Weighted_accuracy_healthy_only_balanced will be assigned nan.")
        Weighted_accuracy_healthy_only_balanced = math.nan

    print("Weighted accuracy balanced Dyslexic only")
    if y_true_dyslexic_only!=[] and y_pred_dyslexic_only!=[]:
        Weighted_accuracy_dyslexic_only_balanced = compute_balanced_accuracy(y_true_dyslexic_only, y_pred_dyslexic_only)
    else:
        print("No dyslexic users found. Weighted_accuracy_dyslexic_only_balanced will be assigned nan.")
        Weighted_accuracy_dyslexic_only_balanced = math.nan

    return Weighted_accuracy, Weighted_accuracy_balanced, Weighted_accuracy_healthy_only, Weighted_accuracy_healthy_only_balanced, Weighted_accuracy_dyslexic_only, Weighted_accuracy_dyslexic_only_balanced


def get_dyslexia_dual_eye_dataset(use_eye_config, arguments):

    if use_eye_config == 'LEFT' \
            or use_eye_config == 'RIGHT' \
            or use_eye_config == 'BOTH_AVG_GAZE':
        ds = Dyslexia(user_limit=arguments.user_limit,                # user_limit argument : to take specific number of users
                      seed=arguments.seed,
                      training_data=arguments.training_data_ratio,
                      use_eye_config=use_eye_config,
                      load_record_duration=arguments.load_record_duration,
                      user_specific_numbers=arguments.user_specific_numbers)   # user_specific_numbers argument : to choose specific mix group of users [DM, DF, HM, HF] e.g. [9 9 9 9]
        dsl = Dyslexia()
        dsr = Dyslexia()
    elif use_eye_config == 'BOTH_AVG_FEAT' \
            or use_eye_config == 'BOTH_COMBINE_FEAT':
        dsl = Dyslexia(user_limit=arguments.user_limit,
                       seed=arguments.seed,
                       training_data=arguments.training_data_ratio,
                       use_eye_config='LEFT',
                       load_record_duration=arguments.load_record_duration,
                       user_specific_numbers=arguments.user_specific_numbers)
        dsr = Dyslexia(user_limit=arguments.user_limit,
                       seed=arguments.seed,
                       training_data=arguments.training_data_ratio,
                       use_eye_config='RIGHT',
                       load_record_duration=arguments.load_record_duration,
                       user_specific_numbers=arguments.user_specific_numbers)
        if use_eye_config == 'BOTH_AVG_FEAT':
            ds = Dyslexia(user_limit=arguments.user_limit,
                          seed=arguments.seed,
                          training_data=arguments.training_data_ratio,
                          use_eye_config='BOTH_AVG_GAZE',
                          load_record_duration=arguments.load_record_duration,
                          user_specific_numbers=arguments.user_specific_numbers)
            # in BOTH_AVG_FEAT and BOTH_AVG_GAZE the input ds will take the AVG gaze points to use it in evaluation.
        elif use_eye_config == 'BOTH_COMBINE_FEAT':
            ds = Dyslexia(user_limit=arguments.user_limit,
                          seed=arguments.seed,
                          training_data=arguments.training_data_ratio,
                          use_eye_config='BOTH_COMBINE_FEAT',
                          load_record_duration=arguments.load_record_duration,
                          user_specific_numbers=arguments.user_specific_numbers)
        else:
            raise Exception('ds was not defined. Maybe something is wrong?')
    else:
        raise Exception('I think this is not a suggested selection.')

    if args.male_ratio:
        ds.male_ratio = args.male_ratio  # gender ratio m/(m+f)
        dsl.male_ratio = args.male_ratio
        dsr.male_ratio = args.male_ratio

    return ds, dsl, dsr

def set_clf_params(args):
    args_new = copy.deepcopy(args)
    if args.classifier == 'logReg':
        args_new.use_normalization = True
    elif args.classifier== 'svm':
        args_new.use_normalization = True
        args_new.svm_c = 0.1
        args_new.svm_gamma = 0.1
    elif args.classifier== 'rbfn':
        args_new.use_normalization = False
        args_new.rbfn_k = 2
    elif args.classifier== 'naive_bayes':
        args_new.use_normalization = True
    elif args.classifier== 'rf':
        args_new.rf_n_estimators= 200
        args_new.use_normalization = True
        args_new.max_depth = 10
        args_new.min_samples_leaf = 1
        args_new.min_samples_split = 2

    return args_new

def main(args):

    ###############
    # PREPARATION #
    ###############
    folder_attachment = '_{clf}_{method}'.format(clf=args.classifier, method=args.method)
    start_logging('EvaluationGender', folder_attachment=folder_attachment)
    logger = logging.getLogger(__name__)
    logger.info('Start Evaluation Gender with Parameters:\n{}'.format(args))
    time_start = time.time()
    ###############

    # dataset selection
    # For classifier 1: dyslexia prediction
    ds, dsl, dsr = get_dyslexia_dual_eye_dataset(args.use_eye_config, args)

    # initialize the right classifier
    ##classifier_list = ['rf', 'rbfn', 'svm', 'svm_linear', 'naive_bayes', 'logReg']

    # For classifier 1
    args.classifier = 'rbfn'
    args1 = set_clf_params(args)
    clf1 = get_classifier(args1.classifier, args1)

    args1.feature_list = "dyslexia_prediction_features"
    args1.feature_number = 8
    args1.label = "label_by_dataset"
    eva = get_method(args1.method, clf1, args1, dataset_name=ds.dataset_name)
    print("args1.feature_list ", args1.feature_list)
    # For classifier 2
    args.classifier = 'logReg'
    args2 = set_clf_params(args)
    clf2 = get_classifier(args2.classifier, args2)
    args2.feature_list = "healthy_gender_prediction_features"
    args2.feature_number = 4
    args2.label = "gender"
    eva_healthy = get_method(args2.method, clf2, args2, dataset_name=ds.dataset_name)
    print("args2.feature_list ", args2.feature_list)

    # For classifier 3
    args.classifier = 'logReg'
    args3 = set_clf_params(args)
    clf3 = get_classifier(args3.classifier, args3)
    args3.feature_list = "dyslexic_gender_prediction_features"
    args3.feature_number = 4
    args3.label = "gender"
    eva_dyslexic = get_method(args3.method, clf3, args3, dataset_name=ds.dataset_name)
    print("args3.feature_list ", args3.feature_list)

    # print("Hi1", args1)
    # print("Hi2", args2)
    # print("Hi3", args3)

    ############
    # LOADING #
    ############
    logger.info('Start Loading (Preparing took {:.3f} seconds).'.format(time.time() - time_start))
    time_loading = time.time()
    ############

    # training
    start_time = datetime.datetime.now()
    # x_train, x_test, y_train, y_test = eva.load_trajectories(ds)
    # print("x_train",x_train[2])
    # print("x_test",x_test[2])

    if args.use_eye_config in ['LEFT', 'RIGHT', 'BOTH_AVG_GAZE']:  # to load the gaze points data for training from load data script
        trajectories_train = ds.load_training_trajectories()
        trajectories_test = ds.load_testing_trajectories()
    elif args.use_eye_config in ['BOTH_AVG_FEAT', 'BOTH_COMBINE_FEAT']: # to load the features data for training
        trajectories_train_xl = dsl.load_training_trajectories()
        trajectories_test_xl = dsl.load_testing_trajectories()
        trajectories_train_xr = dsr.load_training_trajectories()
        trajectories_test_xr = dsr.load_testing_trajectories()
        if args.use_eye_config == 'BOTH_AVG_FEAT':
            # 1-take the average of the features of both eyes possibility
            trajectories_train = (trajectories_train_xl + trajectories_train_xr) / 2
            trajectories_test = (trajectories_test_xl + trajectories_test_xr) / 2
        elif args.use_eye_config == 'BOTH_COMBINE_FEAT':
            # 2-combine the features of both eyes possibility
            trajectories_train = trajectories_train_xl.extend(trajectories_train_xr)
            trajectories_test = trajectories_test_xl.extend(trajectories_test_xr)
        else:
            raise Exception('Should never be')
    else:
        raise Exception('Should never be')
    ## print the user_id list of subjects used in training along with their gender and dyslexia/healthy label
    print('y_train: ', list(zip(trajectories_train.users,
                                trajectories_train.genders,
                                trajectories_train.label_by_datasets)))
    logger.debug('y_train:  {}'.format(list(zip(trajectories_train.users,
                                                trajectories_train.genders,
                                                trajectories_train.label_by_datasets))))
    logger.info('Ratio M({males}) to F({females}): {ratio:.1f}% Male'.format(
        females=trajectories_train.genders.count('F'),
        males=trajectories_train.genders.count('M'),
        ratio=trajectories_train.genders.count('M') / len(trajectories_train) * 100))  ##  gender ratio m/(m+f)

    # print the user_id list of subjects used in testing along with their gender and dyslexia/healthy label
    print('y_test: ', list(zip(trajectories_test.users,       ## to load users
                               trajectories_test.genders,     ## genders argument to predict gender [M, F]
                               trajectories_test.label_by_datasets  ## label_by_datasets argument to predict dyslexia [H, D] or to predict 4 classes [DM, DF, HM, HF]
                           )))
    logger.debug('y_test:  {}'.format(list(zip(trajectories_test.users,
                                               trajectories_test.genders,
                                               trajectories_test.label_by_datasets))))
    logger.info('Ratio M({males}) to F({females})): {ratio:.1f}% Male'.format(
        females=trajectories_test.genders.count('F'),
        males=trajectories_test.genders.count('M'),
        ratio=trajectories_test.genders.count('M') / len(trajectories_test) * 100))
    print('eye_config: ', args.use_eye_config)

    # Data preprocessing for classifier_1 i.e. coordinate transformation and filtering for training data
    trajectories_train = eva.do_data_preparation(trajectories_train)
    logger.debug('Processing of Training: {}'.format(trajectories_train.processing))

    # Data preprocessing i.e. coordinate transformation and filtering for testing data
    trajectories_test = eva.do_data_preparation(trajectories_test)
    logger.debug('Processing of Testing: {}'.format(trajectories_test.processing))

    # NOTE: eva.do_data_preparation(traj) step is not needed for classifier 2 and 3 separately
    # because the dataset is the same so we just proceed with the preprocessed data. We will just extract the
    # useful data needed for classifiers 2 and 3 from the preprocessed trajectories.

    H_D_train_user_num = args.H_D_train_user_num  ##[DM, DF, HM, HF]
    num_healthy_males = H_D_train_user_num[2]
    num_healthy_females = H_D_train_user_num[3]
    num_dyslexic_males = H_D_train_user_num[0]
    num_dyslexic_females = H_D_train_user_num[1]

    # Preparing the pre-processed data for classifier_2
    print("No. of healthy males in trajectories_train: ", len([u for u in trajectories_train.users if u.endswith('3')]))
    print("No. of healthy females in trajectories_train: ", len([u for u in trajectories_train.users if u.endswith('4')]))
    healthy_user_id = random.sample([u for u in trajectories_train.users if u.endswith('3')], num_healthy_males) + random.sample([u for u in trajectories_train.users if u.endswith('4')], num_healthy_females)
    print("Randomly selected healthy subjects: ", healthy_user_id)

    # Preparing the pre-processed data for classifier_3
    print("No. of dyslexic males in trajectories_train: ", len([u for u in trajectories_train.users if u.endswith('1')]))
    print("No. of dyslexic females in trajectories_train: ", len([u for u in trajectories_train.users if u.endswith('2')]))
    dyslexic_user_id = random.sample([u for u in trajectories_train.users if u.endswith('1')], num_dyslexic_males) + random.sample([u for u in trajectories_train.users if u.endswith('2')], num_dyslexic_females)
    print("Randomly selected dyslexic subjects: ", dyslexic_user_id)


    # create a copy of the pre-processed data from selected mixed users to use it in clf2
    trajectories_train_healthy = copy.deepcopy(trajectories_train)
    # Delete the user trajectories that do not belong to healthy subjects
    i = len(trajectories_train_healthy)
    while len(trajectories_train_healthy)!=len(healthy_user_id):
        i -= 1
        if trajectories_train_healthy[i].user not in healthy_user_id:
            trajectories_train_healthy.drop(i)
    # print("Final healthy subjects list: ", trajectories_train_healthy.users)

    # create a copy of the pre-processed data from selected mixed users to use it in clf3
    trajectories_train_dyslexic = copy.deepcopy(trajectories_train)
    # Delete the user trajectories that do not belong to dyslexic subjects
    i = len(trajectories_train_dyslexic)
    while len(trajectories_train_dyslexic)!=len(dyslexic_user_id):
        i -= 1
        if trajectories_train_dyslexic[i].user not in dyslexic_user_id:
            trajectories_train_dyslexic.drop(i)
    # print("Final dyslexic subjects list: ", trajectories_train_dyslexic.users)
    # Now, trajectories_train_healthy stores the pre-processed trajectories only for healthy subjects selected by two arguments num_healthy_males and num_healthy_females

    ############
    # ANALYSIS #
    ############
    extract_time = time.time()

    logger.info('Start Extraction (Loading test and train data took {:.3f} seconds).'.format(
        time.time() - time_loading))

    # Number of training subjects = training ratio * total number of users (defined by user_specific_numbers)
    print("Feature extraction for {} training cases (classifier 1)".format(len(trajectories_train)))
    labeled_feature_values_train = eva.provide_feature(trajectories=trajectories_train,
                                                       normalize=args1.use_normalization,
                                                       label=args1.label)
    # Number of test subjects = (1 - training ratio) * total number of users (defined by user_specific_numbers)
    print("Feature extraction for {} testing cases (classifier 1)".format(len(trajectories_test)))
    labeled_feature_values_test = eva.provide_feature(trajectories=trajectories_test,
                                                      normalize=args1.use_normalization,
                                                      label=args1.label)

    # Feature Extraction for classifier 2 (gender prediction in healthy subjects)
    # Number of training subjects = training ratio * total number of users (defined by user_specific_numbers)
    print("Feature extraction for {} training cases for healthy subjects (classifier 2)".format(len(trajectories_train_healthy)))
    labeled_feature_values_train_healthy = eva_healthy.provide_feature(trajectories=trajectories_train_healthy,
                                                       normalize=args2.use_normalization,
                                                       label="gender")

    # Number of test subjects = (1 - training ratio) * total number of users (defined by user_specific_numbers)
    print("Feature extraction for {} testing cases (classifier 2)".format(len(trajectories_test)))
    labeled_feature_values_test_healthy = eva_healthy.provide_feature(trajectories=trajectories_test,
                                                      normalize=args2.use_normalization,
                                                      label="gender")

    # Feature Extraction for classifier 3 (gender prediction in dyslexic subjects)
    # Number of training subjects = training ratio * total number of users (defined by user_specific_numbers)
    print("Feature extraction for {} training cases for dyslexic subjects (classifier 3)".format(len(trajectories_train_dyslexic)))
    labeled_feature_values_train_dyslexic = eva_dyslexic.provide_feature(trajectories=trajectories_train_dyslexic,
                                                       normalize=args3.use_normalization,
                                                       label="gender")

    # Number of test subjects = (1 - training ratio) * total number of users (defined by user_specific_numbers)
    print("Feature extraction for {} testing cases (classifier 3)".format(len(trajectories_test)))
    labeled_feature_values_test_dyslexic = eva_dyslexic.provide_feature(trajectories=trajectories_test,
                                                      normalize=args3.use_normalization,
                                                      label="gender")



    if args1.use_normalization:
        for jj in range(len(labeled_feature_values_train)):
            labeled_feature_values_train[jj].reset_index(drop=True, inplace=True)  # just to reset the index
            labeled_feature_values_test[jj].reset_index(drop=True, inplace=True)

            # scaler = MinMaxScaler()
            scaler = StandardScaler()  # the range of the data value

            train_columns_names=  labeled_feature_values_train[jj].select_dtypes(exclude=['object']).columns
            test_columns_names= labeled_feature_values_test[jj].select_dtypes(exclude=['object']).columns

            logger.debug('normalizing')

            train_normalize_data = scaler.fit_transform(labeled_feature_values_train[jj].select_dtypes(exclude=['object']))
            test_normalize_data = scaler.transform(labeled_feature_values_test[jj].select_dtypes(exclude=['object']))
            labeled_feature_values_train[jj] = pd.concat([pd.DataFrame(train_normalize_data,columns=train_columns_names), labeled_feature_values_train[jj].select_dtypes(exclude=['int', 'float64'])], axis=1)
            labeled_feature_values_test[jj] = pd.concat([pd.DataFrame(test_normalize_data , columns=test_columns_names), labeled_feature_values_test[jj].select_dtypes(exclude=['int', 'float64'])], axis=1)

    # Normalization of classifier 2
    if args2.use_normalization:
        for jj in range(len(labeled_feature_values_train_healthy)):
            labeled_feature_values_train_healthy[jj].reset_index(drop=True, inplace=True)  # just to reset the index
            labeled_feature_values_test_healthy[jj].reset_index(drop=True, inplace=True)

            # scaler = MinMaxScaler()
            scaler = StandardScaler()  # the range of the data value

            train_columns_names=  labeled_feature_values_train_healthy[jj].select_dtypes(exclude=['object']).columns
            test_columns_names= labeled_feature_values_test_healthy[jj].select_dtypes(exclude=['object']).columns

            logger.debug('normalizing')

            train_normalize_data = scaler.fit_transform(labeled_feature_values_train_healthy[jj].select_dtypes(exclude=['object']))
            test_normalize_data = scaler.transform(labeled_feature_values_test_healthy[jj].select_dtypes(exclude=['object']))
            labeled_feature_values_train_healthy[jj] = pd.concat([pd.DataFrame(train_normalize_data,columns=train_columns_names), labeled_feature_values_train_healthy[jj].select_dtypes(exclude=['int', 'float64'])], axis=1)
            labeled_feature_values_test_healthy[jj] = pd.concat([pd.DataFrame(test_normalize_data , columns=test_columns_names), labeled_feature_values_test_healthy[jj].select_dtypes(exclude=['int', 'float64'])], axis=1)

    # Normalization of classifier 3
    if args3.use_normalization:
        for jj in range(len(labeled_feature_values_train_dyslexic)):
            labeled_feature_values_train_dyslexic[jj].reset_index(drop=True, inplace=True)  # just to reset the index
            labeled_feature_values_test_dyslexic[jj].reset_index(drop=True, inplace=True)

            # scaler = MinMaxScaler()
            scaler = StandardScaler()  # the range of the data value

            train_columns_names=  labeled_feature_values_train_dyslexic[jj].select_dtypes(exclude=['object']).columns
            test_columns_names= labeled_feature_values_test_dyslexic[jj].select_dtypes(exclude=['object']).columns

            logger.debug('normalizing')
            train_normalize_data = scaler.fit_transform(labeled_feature_values_train_dyslexic[jj].select_dtypes(exclude=['object']))
            test_normalize_data = scaler.transform(labeled_feature_values_test_dyslexic[jj].select_dtypes(exclude=['object']))
            labeled_feature_values_train_dyslexic[jj] = pd.concat([pd.DataFrame(train_normalize_data,columns=train_columns_names), labeled_feature_values_train_dyslexic[jj].select_dtypes(exclude=['int', 'float64'])], axis=1)
            labeled_feature_values_test_dyslexic[jj] = pd.concat([pd.DataFrame(test_normalize_data , columns=test_columns_names), labeled_feature_values_test_dyslexic[jj].select_dtypes(exclude=['int', 'float64'])], axis=1)

#################################################################
    ############
    # TRAINING #
    ############
    time_training = time.time()
    logger.info('Start Training (Feature Extraction took {:.3f} seconds).'.format(time.time() - extract_time))
    ############

    # !!Main training step!!
    # For training the classifier 1
    eva.train(labeled_feature_values=labeled_feature_values_train)
    # For training the classifier 2
    eva_healthy.train(labeled_feature_values=labeled_feature_values_train_healthy)
    # For training the classifier 3
    eva_dyslexic.train(labeled_feature_values=labeled_feature_values_train_dyslexic)

    ##############
    # TESTING #  # eva is get_method
    ##############
    time_evaluating = time.time()
    logger.info('Start Evaluating (Training took {:.3f} seconds).'.format(time.time() - time_training))
    ##############

    # Print the accuracy and balanced accuracy (max. a-priori) of training data to check the overfitting along with true vs predicted labels
    print('### Evaluating Train-Data (overfitting check) ###')
    logger.info('Evaluating Train-Data (overfitting check)')
    res_train = eva.evaluation(feature_values=labeled_feature_values_train)

    print('### Evaluating Train-Data (overfitting check) ###')
    logger.info('Evaluating Train-Data (overfitting check)')
    res_train_healthy = eva_healthy.evaluation(feature_values=labeled_feature_values_train_healthy)

    print('### Evaluating Train-Data (overfitting check) ###')
    logger.info('Evaluating Train-Data (overfitting check)')
    res_train_dyslexic = eva_dyslexic.evaluation(feature_values=labeled_feature_values_train_dyslexic)

    # Print the accuracy and balanced accuracy (max. a-priori) of test data to check the overfitting along with true vs predicted labels
    print('### Evaluating Test-Data ###')
    logger.info('Evaluating Test-Data')
    res = eva.evaluation(feature_values=labeled_feature_values_test)         #!!Main Testing Step!!
    res_healthy = eva_healthy.evaluation(feature_values=labeled_feature_values_test_healthy)         #!!Main Testing Step!!
    res_dyslexic = eva_dyslexic.evaluation(feature_values=labeled_feature_values_test_dyslexic)         #!!Main Testing Step!!

    print('###')
    print('### Result EvaGender {method} {clf} for {user_train}+{user_test} user'.format(
        method=args1.method, clf=args1.classifier, user_train=len(trajectories_train),
        user_test=len(trajectories_test)))
    print('###')
    print('### Accuracy Train CLF1: {acc} ({acc_bal} balanced)'.format(
        acc=res_train['Accuracy'], acc_bal=res_train['Accuracy_balanced']))
    print('### Accuracy Test  CLF1: {acc} ({acc_bal} balanced)'.format(
        acc=res['Accuracy'], acc_bal=res['Accuracy_balanced']))
    print('###')

    print('### Result EvaGender Classifier 2 {method} {clf} for {user_train} user'.format(
        method=args2.method, clf=args2.classifier, user_train=len(trajectories_train_healthy)))
    print('###')
    print('### Accuracy Train  CLF2: {acc} ({acc_bal} balanced)'.format(
        acc=res_train_healthy['Accuracy'], acc_bal=res_train_healthy['Accuracy_balanced']))
    print('### Accuracy Test  CLF2: {acc} ({acc_bal} balanced)'.format(
        acc=res_healthy['Accuracy'], acc_bal=res_healthy['Accuracy_balanced']))
    print('###')

    print('### Result EvaGender Classifier 3 {method} {clf} for {user_train} user'.format(
        method=args3.method, clf=args3.classifier, user_train=len(trajectories_train_dyslexic)))
    print('###')
    print('### Accuracy Train  CLF3: {acc} ({acc_bal} balanced)'.format(
        acc=res_train_dyslexic['Accuracy'], acc_bal=res_train_dyslexic['Accuracy_balanced']))
    print('### Accuracy Test  CLF3: {acc} ({acc_bal} balanced)'.format(
        acc=res_dyslexic['Accuracy'], acc_bal=res_dyslexic['Accuracy_balanced']))
    print('###')

    # Voting based hierarchical classifier
    voting_accuracy, voting_accuracy_balanced, voting_accuracy_healthy_only, voting_accuracy_healthy_only_balanced, voting_accuracy_dyslexic_only, voting_accuracy_dyslexic_only_balanced = voting_based_classifier(res, res_healthy, res_dyslexic)
    # Weighting based hierarchical classifier
    Weighted_accuracy, Weighted_accuracy_balanced, Weighted_accuracy_healthy_only, Weighted_accuracy_healthy_only_balanced, Weighted_accuracy_dyslexic_only, Weighted_accuracy_dyslexic_only_balanced = weighting_based_classifier(res, res_healthy, res_dyslexic)

    # delete the unncessary keys("dyslexic","male") that we added to res for computing voting_res and weighting_res
    if 'dyslexic' in res: del res['dyslexic']
    if 'healthy' in res: del res['healthy']

    # save the voting_res instead of the first classifier accuracy ?? is there better way?
    res['voting_accuracy'] = voting_accuracy
    res['voting_accuracy_balanced'] = voting_accuracy_balanced
    res['voting_accuracy_healthy_only'] = voting_accuracy_healthy_only
    res['voting_accuracy_healthy_only_balanced'] = voting_accuracy_healthy_only_balanced
    res['voting_accuracy_dyslexic_only'] = voting_accuracy_dyslexic_only
    res['voting_accuracy_dyslexic_only_balanced'] = voting_accuracy_dyslexic_only_balanced

    res['Weighted_accuracy'] = Weighted_accuracy
    res['Weighted_accuracy_balanced'] = Weighted_accuracy_balanced
    res['Weighted_accuracy_healthy_only'] = Weighted_accuracy_healthy_only
    res['Weighted_accuracy_healthy_only_balanced'] = Weighted_accuracy_healthy_only_balanced
    res['Weighted_accuracy_dyslexic_only'] = Weighted_accuracy_dyslexic_only
    res['Weighted_accuracy_dyslexic_only_balanced'] = Weighted_accuracy_dyslexic_only_balanced


    res['Train_Accuracy_clf1'] = res_train['Accuracy']
    res['Train_Accuracy_Balanced_clf1'] = res_train['Accuracy_balanced']

    res['Train_Accuracy_clf2'] = res_train_healthy['Accuracy']
    res['Train_Accuracy_Balanced_clf2'] = res_train_healthy['Accuracy_balanced']

    res['Train_Accuracy_clf3'] = res_train_dyslexic['Accuracy']
    res['Train_Accuracy_Balanced_clf3'] = res_train_dyslexic['Accuracy_balanced']

    # saving results
    res['config'] = vars(args)    ## we put all the config parameters to save them in JSON result files
    res['config']['classifier'] = args1.classifier + '_' +args2.classifier + '_' + args3.classifier
    res['config']['feature_number'] = str(args1.feature_number) + '_' + str(args2.feature_number) + '_' + str(args3.feature_number)
    ##print("iam here", res['config']['classifier'])

    result_name = '&'.join([f'{k}={v}' for k, v in res['config'].items()])
    res['params'] = result_name

    # store the results as the below steucture in results folder

    filename = './results/Heirarchical_classifier_with' + str(args.user_specific_numbers) + 'and' + str(args.H_D_train_user_num) +  '/featuers_score_' +\
               'MD{MD}+FD{FD}+MH{MH}+FH{FH}/'. format(MD=args.user_specific_numbers[0],
                                                      FD=args.user_specific_numbers[1],
                                                      MH=args.user_specific_numbers[2],
                                                      FH=args.user_specific_numbers[3]) \
               + args.use_eye_config + '/' \
               + args.method + '/' \
               + args1.classifier + '_' + args2.classifier + '_' + args3.classifier + '/' \
               + 'IVT_' + str(args.ivt_threshold) + '/' \
               + 'FEATS_' + str(args1.feature_number) + '_' + str(args2.feature_number) + '_' + str(args3.feature_number) + '/' \
               + '/seed_' + str(args.seed) + '_' \
               + datetime.datetime.strftime(datetime.datetime.now(),'%y-%m-%d_%H:%M:%S') + '.json'

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as outfile:
        json.dump(res, outfile)

    ############
    # FINISHED #
    ############
    logger.info(
        'Start Training (Loading test and train data took {:.3f} seconds).'.format(time.time() - time_evaluating))
    logger.info('Complete Evaluation Gender took {:.3f} seconds.'.format(time.time() - time_start))
    ############
    print("total time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))



def add_parser_arguments(parser):

    # optional parameters - used depending on the method
    parser.add_argument('--rbfn_k', type=int, default=32)

    parser.add_argument('--class_weight', default='balanced')

    parser.add_argument('--ivt_threshold', type=int, default=50,
                        help='Velocity threshold for ivt.')

    parser.add_argument('--load_record_duration', type=int, default=0,
                        help='Can be used to load part of the data record')

    parser.add_argument('--train_using_RFE_top_features', type=int, nargs='*',
                        help='Set flag', default=0)


    # dataset (copied to general handling)
    parser.add_argument('--male_ratio', type=float, default=0,
                        help='When user_limit is given, we try to return the given male_ratio.'
                             'If Ratio is 0, than return ratio from Dataset.')

    parser.add_argument('--training_data_ratio', type=float, default=0.8,
                        help='ratio of train samples to all samples. The rest will be used for test.'
                             'Only to be used with Dyslexia dataset')
    parser.add_argument('--user_specific_numbers', type=int, nargs=4, default=[0, 0, 0, 0],
                        help='HARDCODE for Dyslecia: Select Specific number of users:'
                             'Male Dyslexic, Female Dyslexic, Male Healthy, Female Healthy')

    parser.add_argument('--H_D_train_user_num', type=int, nargs=4, default=[0, 0, 0, 0],
                        help='Select Specific number of users to train clf2 and clf3:'
                             'Male Dyslexic, Female Dyslexic, Male Healthy, Female Healthy')
    return parser

if __name__ == '__main__':

    # get parser with default parameter
    parser = get_default_parser()
    parser.add_argument('-ds', '--dataset', default='dyslexia', required=False)
    parser.add_argument('-mf', '--modelfile', required=False)
    parser.add_argument('--label', choices=['gender', 'label_by_dataset'], default='gender', required=False)
    # check them and get new parser
    args, _ = parser.parse_known_args()
    parser = get_conditional_parser(parser, args)
    parser.add_argument('-ds', '--dataset', default='dyslexia', required=False)  # necessary again

    parser = add_parser_arguments(parser)

    # pars arguments
    args = parser.parse_args()

    print(args)

    main(args)
