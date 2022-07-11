""" This method should provide flexible feature. It based on evaluation_our_append."""
import datetime
import os

import numpy as np

import sklearn

from schau_mir_in_die_augen.evaluation.base_evaluation import BaseEvaluation, FeatureLabels
from schau_mir_in_die_augen.features import BasicFeatures1D, BasicFeatures2D, ExtendedFeatures2D, \
    TimeFeatures, HistoryFeatures
from schau_mir_in_die_augen.feature_extraction import Data, Dimension, ComplexFeatures, plot_dt


class EvaluationGeneralFixSac(BaseEvaluation):

    def __init__(self, base_clf=None,
                 vel_threshold=50, min_fix_duration=.1):

        super().__init__()
        self.min_fix_duration = min_fix_duration
        self.vel_threshold = vel_threshold
        self.feature_sac = []
        self.feature_fix = []
        if base_clf is None:
            self.clf_fix = []
            self.clf_sac = []
        else:
            self.clf_fix = sklearn.base.clone(base_clf)
            self.clf_sac = sklearn.base.clone(base_clf)
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.graph_path = '{}/../../data/'.format(file_dir)

    def get_split_parameter(self):
        """ Return specific Parameter for split

        e.g. IVT with velocity threshold 50 deg/s and minimal fixation duration of 0.1 seconds
        """
        split_method = 'IVT'
        split_parameter = {'vel_threshold': self.vel_threshold, 'min_fix_duration': self.min_fix_duration}
        clf_parameter = {}

        return split_method, split_parameter, clf_parameter

    def train(self, labeled_feature_values):
        feature_values, feature_labels = self.separate_feature_labels(labeled_feature_values)

        # Adding 2 lines for feature column names to the class variable to be used when plotting Decision tree
        self.feature_sac = feature_values[0].columns.array
        self.feature_fix = feature_values[1].columns.array

        # if top features are given it will reduce features
        feature_values = self.select_top_features(feature_values)

        print("X_train_sac", feature_values[0].shape)
        print("X_train_fix", feature_values[1].shape)

        print("Training")
        start_time = datetime.datetime.now()
        self.clf_sac.fit(feature_values[0], feature_labels[0])
        self.clf_fix.fit(feature_values[1], feature_labels[1])

        print("train time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))

    def get_classifiers_and_names(self):
        return [self.clf_sac, self.clf_fix], ['sac', 'fix']

    def plot_decision_tree(self, features: str, index: int, target_data: np.ndarray):
        """ To plot a single decision tree from the trained RDF model.

        Will plot 2 decision trees if 2 models are trained for saccade and fixations

        :param features: a string denoting whether all features or top n features where used for training RDF model
        :param index: A value to plot a tree on given index
        :param target_data: A array of labels used for trainnig the RDF mocel
        """

        if self.__class__.__name__ == 'OurEvaluationOne' or self.__class__.__name__ == 'EvaluationWindowed':
            if features == 'all':   # todo: this should be unified
                feature_col = self.feature
            else:
                feature_col = self.n_top_features[0]
            estimator = self.clf.estimators_[index]
            # this is the position of the tree so if our estimator is 200 means we have(0 to 199 index)
            # for estimator in self.clf_sac.estimators_:  #this loop in case we want to draw all the trees
            plot_dt(estimator, self.graph_path, feature_col, target_data, self.__class__.__name__)
            # filename will be the class name of the method used for the prediction
        else:
            if features == 'all':
                feature_col = [self.feature_sac, self.feature_fix]
            else:
                feature_col = self.n_top_features

            estimator = self.clf_sac.estimators_[index]
            plot_dt(estimator, self.graph_path, feature_col[0], target_data, self.__class__.__name__+'_sac')

            estimator = self.clf_fix.estimators_[index]
            plot_dt(estimator, self.graph_path, feature_col[1], target_data, self.__class__.__name__+'_fix')


class EvaluationGeneralFixSacNew(EvaluationGeneralFixSac):

    def __init__(self, base_clf=None,
                 vel_threshold=50, min_fix_duration=.1,
                 name_feature_set: (str, list) = None,
                 name_second_feature_set: (str, list) = None):

        super().__init__(base_clf=base_clf, min_fix_duration=min_fix_duration, vel_threshold=vel_threshold)

        self.feature_entries = self.get_feature_entries(name_feature_set)
        self.second_feature_entries = self.get_second_feature_entries(name_second_feature_set)

    def get_split_parameter(self):
        """ Return specific Parameter for split
        e.g. IVT with velocity threshold 50 deg/s and minimal fixation duration of 0.1 seconds
        """
        split_method = 'IVT'
        split_parameter = {'vel_threshold': self.vel_threshold, 'min_fix_duration': self.min_fix_duration}
        clf_parameter = {'minimum_sample_length': 4}

        return split_method, split_parameter, clf_parameter

    def trajectory_split_and_feature(self, trajectory):
        """ Generate feature vectors for all saccades and fixations in a trajectory

        :param trajectory: ndarray
            2D array of gaze points (x,y)
        :return: list, list
            list of feature vectors (each is a 1D ndarray)
        """

        # get feature dataFrame
        features = self.trajectory_split_and_feature_basic(trajectory=trajectory, feature_entries=self.feature_entries)

        # separate saccades and fixations and get feature data
        features_sacc = features[(features['sample_type'] == 'saccade') & features['data_data_duration'] > 0.012]
        features_fix = features[features['sample_type'] == 'fixation']
        # remove 'is_saccade' column
        features_sacc.drop(['sample_type'], axis=1, inplace=True)
        features_fix.drop(['sample_type'], axis=1, inplace=True)

        return [features_sacc, features_fix]

    @staticmethod
    def get_feature_entries(name_feature_set: (str, list) = None):
        """ This method provides different feature sets.
            You can combine sets!
        """

        if name_feature_set is None:
            return EvaluationGeneralFixSacNew.get_feature_entries('default')
        elif isinstance(name_feature_set, str):
            if name_feature_set == 'default':
                name_feature_set = ['basic', 'angular_test']
            else:
                name_feature_set = [name_feature_set]
        elif not isinstance(name_feature_set, list):
            raise Exception('Expected string or list for name_feature_set. Got {}'.format(type(name_feature_set)))

        feature_entries = set()

        for feature_set in name_feature_set:
            if feature_set == 'angular_test':
                """ Testing with only angular features.
                This set is not complete
                """

                feature_entries = feature_entries | {
                    *[(data, dimension, feature)  # basic statistical feature
                      for data in [Data.deg_data, Data.deg_velocity, Data.deg_acceleration]
                      for dimension in [Dimension.x_data, Dimension.y_data, Dimension.xy_length]
                      for feature in [BasicFeatures1D.mean,
                                      BasicFeatures1D.max,
                                      BasicFeatures1D.min]]  # todo add more
                }
            elif feature_set == 'basic':
                """ Really basic features """
                feature_entries = feature_entries | {
                    (Data.data, Dimension.data, BasicFeatures1D.len),  # length of subset
                    (Data.data, Dimension.data, TimeFeatures.duration),  # duration (length / sample_rate)
                }
            else:
                raise Exception('Set {} is unknown'.format(feature_set))

        return feature_entries

    @staticmethod
    def get_second_feature_entries(name_feature_set: (str, list) = None):
        """ This method provides different feature sets.
            You can combine sets!
        """

        if name_feature_set is None:
            return []
        elif isinstance(name_feature_set, str):
            if name_feature_set == 'default':
                name_feature_set = ['basic']
            else:
                name_feature_set = [name_feature_set]
        elif not isinstance(name_feature_set, list):
            raise Exception('Expected string or list for name_feature_set. Got {}'.format(type(name_feature_set)))

        feature_entries = []

        for feature_set in name_feature_set:
            if feature_set == 'basic':
                """ Really basic features """
                feature_entries += ['number_subsets']
            else:
                raise Exception('Set {} is unknown'.format(feature_set))

        return feature_entries

    # noinspection PyMethodMayBeStatic
    def get_second_level_features(self, labeled_feature_values):
        """ add other features columns depending on all feature rows"""

        for clf_id in range(len(labeled_feature_values)):

            if 'number_subsets' in self.second_feature_entries:
                labeled_feature_values[clf_id]['number_subsets'] = None

            for user in list(set(labeled_feature_values[clf_id][FeatureLabels.user])):
                for case in list(set(labeled_feature_values[clf_id][FeatureLabels.case])):

                    selection = (labeled_feature_values[clf_id][FeatureLabels.user] == user) \
                                & (labeled_feature_values[clf_id][FeatureLabels.case] == case)

                    if 'number_subsets' in self.second_feature_entries:
                        # add the number of subsets
                        labeled_feature_values[clf_id].loc[selection, ['number_subsets']] = \
                            labeled_feature_values[clf_id].loc[selection].shape[0]

        return labeled_feature_values
