import os
import sys
import logging
import datetime
import json
import time
import pandas as pd

from sklearn.preprocessing import StandardScaler
sys.path.append('../')

from schau_mir_in_die_augen.evaluation.base_selection import \
    get_method, get_classifier, get_default_parser, get_conditional_parser, start_logging
from schau_mir_in_die_augen.datasets.Dyslexia import Dyslexia
from schau_mir_in_die_augen.evaluation.base_evaluation import FeatureLabels

def get_dyslexia_dual_eye_dataset(use_eye_config, arguments):

    # todo: these should be in Dyslexia dataset class?

    if use_eye_config == 'LEFT' \
            or use_eye_config == 'RIGHT' \
            or use_eye_config == 'BOTH_AVG_GAZE':
        ds = Dyslexia(user_limit=arguments.user_limit,
                      seed=arguments.seed,
                      training_data=arguments.training_data_ratio,
                      use_eye_config=use_eye_config,
                      load_record_duration=arguments.load_record_duration,
                      user_specific_numbers=arguments.user_specific_numbers)
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
        ds.male_ratio = args.male_ratio
        dsl.male_ratio = args.male_ratio
        dsr.male_ratio = args.male_ratio

    return ds, dsl, dsr

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
    ds, dsl, dsr = get_dyslexia_dual_eye_dataset(args.use_eye_config, args)

    # initialize the right classifier
    clf = get_classifier(args.classifier, args)

    args.dataset = 'tex'  # to use tex features
    eva = get_method(args.method, clf, args, dataset_name=ds.dataset_name)

    ############
    # LOADING #
    ############
    logger.info('Start Loading (Preparing took {:.3f} seconds).'.format(time.time() - time_start))
    time_loading = time.time()
    ############

    # training
    start_time = datetime.datetime.now()
    # todo: these should maybe be done inside the dataset
    # TODO: whats the difference between BOTH_AVG_GAZE and BOTH_AVG_FEAT !?
    if args.use_eye_config in ['LEFT', 'RIGHT', 'BOTH_AVG_GAZE']:
        trajectories_train = ds.load_training_trajectories()
        trajectories_test = ds.load_testing_trajectories()
    elif args.use_eye_config in ['BOTH_AVG_FEAT', 'BOTH_COMBINE_FEAT']:
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

    print('y_train: ', list(zip(trajectories_train.users,
                                trajectories_train.genders,
                                trajectories_train.label_by_datasets)))
    logger.debug('y_train:  {}'.format(list(zip(trajectories_train.users,
                                                trajectories_train.genders,
                                                trajectories_train.label_by_datasets))))
    logger.info('Ratio M({males}) to F({females}): {ratio:.1f}% Male'.format(
        females=trajectories_train.genders.count('F'),
        males=trajectories_train.genders.count('M'),
        ratio=trajectories_train.genders.count('M') / len(trajectories_train) * 100))
    print('y_test: ', list(zip(trajectories_test.users,
                               trajectories_test.genders,
                               trajectories_test.label_by_datasets
                           )))
    logger.debug('y_test:  {}'.format(list(zip(trajectories_test.users,
                                               trajectories_test.genders,
                                               trajectories_test.label_by_datasets))))
    logger.info('Ratio M({males}) to F({females})): {ratio:.1f}% Male'.format(
        females=trajectories_test.genders.count('F'),
        males=trajectories_test.genders.count('M'),
        ratio=trajectories_test.genders.count('M') / len(trajectories_test) * 100))
    print('eye_config: ', args.use_eye_config)

    trajectories_train = eva.do_data_preparation(trajectories_train)
    logger.debug('Processing of Training: {}'.format(trajectories_train.processing))

    trajectories_test = eva.do_data_preparation(trajectories_test)
    logger.debug('Processing of Testing: {}'.format(trajectories_test.processing))

    ############
    # ANALYSIS #
    ############
    extract_time = time.time()

    logger.info('Start Extraction (Loading test and train data took {:.3f} seconds).'.format(
        time.time() - time_loading))

    print("Feature extraction for {} training cases".format(len(trajectories_train)))
    labeled_feature_values_train = eva.provide_feature(trajectories=trajectories_train,
                                                       normalize=args.use_normalization,
                                                       label=args.label)

    print("Feature extraction for {} testing cases".format(len(trajectories_test)))
    labeled_feature_values_test = eva.provide_feature(trajectories=trajectories_test,
                                                      normalize=args.use_normalization,
                                                      label=args.label)


 ################################################################ new norm

    if args.use_normalization:
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
            labeled_feature_values_train[jj] = pd.concat([pd.DataFrame(train_normalize_data,columns=train_columns_names), labeled_feature_values_train[jj].select_dtypes(exclude=['int', 'float64'])],axis=1)
            labeled_feature_values_test[jj] = pd.concat([pd.DataFrame(test_normalize_data , columns=test_columns_names), labeled_feature_values_test[jj].select_dtypes(exclude=['int', 'float64'])], axis=1)

#################################################################
    ############
    # TRAINING #
    ############
    time_training = time.time()
    logger.info('Start Training (Feature Extraction took {:.3f} seconds).'.format(time.time() - extract_time))
    ############

    if args.train_top_features:
        # if we do not provide train_top_features argument, this condition become False=0
        # noinspection PyUnboundLocalVariable
        eva.top_features(labeled_feature_values=labeled_feature_values_train, seed=args.seed,
                         n=args.train_top_features)

    if args.train_using_RFE_top_features:
        # if we do not provide train_using_RFE_top_features argument, this condition become False=0
        eva.ref_top_features(feature_values=labeled_feature_values_train, seed=args.seed,
                             n=args.train_using_RFE_top_features)

    eva.train(labeled_feature_values=labeled_feature_values_train)

    # If plot_RDF_tree argument is provided with value > 0,
    # then we will plot a decision tree with the index value provided in this argument
    if clf == 'rf' and args.plot_RDF_tree > 0:
        if len(args.train_top_features) > 0:
            eva.plot_decision_tree('top', args.plot_RDF_tree, labeled_feature_values_train[FeatureLabels.user])
        else:
            eva.plot_decision_tree('all', args.plot_RDF_tree, labeled_feature_values_train[FeatureLabels.user])

    ##############
    # EVALUATING #
    ##############
    time_evaluating = time.time()
    logger.info('Start Evaluating (Training took {:.3f} seconds).'.format(time.time() - time_training))
    ##############

    print('### Evaluating Train-Data (overfitting check) ###')
    logger.info('Evaluating Train-Data (overfitting check)')
    res_train = eva.evaluation(feature_values=labeled_feature_values_train)

    if args.use_optimal_threshold:
        eva.use_optimal_threshold = True
        eva.binary_classifier_threshold = res_train['best_threshold_ROC']    # we use the optimal threshold from ROC curve as threshold for the classifier in test cases (comment it when i do not want to use the optimal threshold)

    print('### Evaluating Test-Data ###')
    logger.info('Evaluating Test-Data')
    res = eva.evaluation(feature_values=labeled_feature_values_test)

    print('###')
    print('### Result EvaGender {method} {clf} for {user_train}+{user_test} user'.format(
        method=args.method, clf=args.classifier, user_train=len(trajectories_train),
        user_test=len(trajectories_test)))
    print('###')
    print('### Accuracy Train: {acc} ({acc_bal} balanced)'.format(
        acc=res_train['Accuracy'], acc_bal=res_train['Accuracy_balanced']))
    print('### Accuracy Test: {acc} ({acc_bal} balanced)'.format(
        acc=res['Accuracy'], acc_bal=res['Accuracy_balanced']))
    print('###')

    # Adding more Information
    #   todo: there could be somhow a general file with all important data.
    res['Train_Accuracy'] = res_train['Accuracy']
    res['Train_Accuracy_Balanced'] = res_train['Accuracy_balanced']

    if args.label == 'label_by_dataset':
        Class1 = 'dyslexic'
        Class2 = 'healthy'
    elif args.label == 'gender':
        Class1 = 'F'
        Class2 = 'M'

    res['Train_True_Lable'] = res_train['True_Lable']
    res['Train_Predicted_Lable'] = res_train['Predicted_Lable']
    res['Train_'+Class1+'_prob'] = res_train[Class1+'_prob']
    res['Train_'+Class2+'_prob'] = res_train[Class2+'_prob']

    res['Train_precision_score']  = res_train['precision_score']
    res['Train_recall_score'] = res_train['recall_score']
    res['Train_f1_score'] = res_train['f1_score']
    res['Train_area_under_ROC_curve'] =res_train['area_under_ROC_curve']
    res['Train_best_threshold_ROC'] = res_train['best_threshold_ROC']

    # saving results
    res['config'] = vars(args)
    result_name = '&'.join([f'{k}={v}' for k, v in res['config'].items()])
    res['params'] = result_name
    if args.train_top_features or args.train_using_RFE_top_features:
        print("Top features for saccade: ", eva.n_top_features[0].to_numpy())
        print("Top features for fixations: ", eva.n_top_features[1].to_numpy())
        res['top_sac_features'] = eva.n_top_features[0].to_numpy().tolist()
        res['top_fix_features'] = eva.n_top_features[1].to_numpy().tolist()

    use_top_feat = 'train_without_top_features'
    if args.train_using_RFE_top_features != 0:
        use_top_feat = 'train_using_RFE_top_features'
    elif args.train_top_features != 0:
        use_top_feat = 'train_using_top_features'

    if args.label == 'label_by_dataset':
        use_label = 'predict_HD'
    else:
        use_label = 'predict_MF'
    # if I want to store the results in gender_eval_res folder
    # # delete the unncessary keys("dyslexic","male") that we added to res for computing voting_res and weighting_res
    if 'dyslexic' in res: del res['dyslexic']
    if 'healthy' in res: del res['healthy']
    if 'M' in res: del res['M']
    if 'F' in res: del res['F']

    filename = './results/Gender_prediction_paper_accuracy/' + use_label + '_with_' + args.feature_list + '_trainRatio_' + str(args.training_data_ratio) + '/featuers_score_' +\
               'MD{MD}+FD{FD}+MH{MH}+FH{FH}/'. format(MD=args.user_specific_numbers[0],
                                                      FD=args.user_specific_numbers[1],
                                                      MH=args.user_specific_numbers[2],
                                                      FH=args.user_specific_numbers[3]) \
               + args.use_eye_config + '/' \
               + args.method + '/' \
               + args.classifier + '/' \
               + 'IVT_' + str(args.ivt_threshold) + '/' \
               + 'FEATS_' + str(args.feature_number) + '/' \
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
    parser.add_argument('--use_optimal_threshold', action='store_true',
                        help='Use optimal threshold from train and use it in the probablities to classify in the test (use it only in case of binary classification problem)')



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
