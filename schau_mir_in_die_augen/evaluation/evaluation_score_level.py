import datetime

from schau_mir_in_die_augen.evaluation.evaluation_general import EvaluationGeneralFixSac
from schau_mir_in_die_augen.feature_extraction import sac_subset, fix_subset


class ScoreLevelEvaluation(EvaluationGeneralFixSac):
    def __init__(self, base_clf,
                 vel_threshold=50, min_fix_duration=.1,
                 text_features=False):

        super().__init__(base_clf=base_clf, min_fix_duration=min_fix_duration, vel_threshold=vel_threshold)

        self.text_features = text_features

    # noinspection PyMethodMayBeStatic
    def get_feature_parameter(self):
        """ Return which features should be calculated. """

        omit_our = True
        omit_stats = False

        return omit_our, omit_stats

    def trajectory_split_and_feature(self, trajectory):
        """ Generate feature vectors for all saccades and fixations in a trajectory

        :param trajectory: ndarray
            2D array of gaze points (x,y)
        :return: list, list
            list of feature vectors (each is a 1D ndarray)
        """

        # get feature dataFrame
        features = self.trajectory_split_and_feature_basic(trajectory)

        # separate saccades and fixations and get feature data
        features_sacc = features[(features['sample_type'] == 'saccade') & features['duration'] > 0.012]
        features_fix = features[features['sample_type'] == 'fixation']

        features_sacc = features_sacc[sac_subset(self.text_features)]
        features_fix = features_fix[fix_subset(self.text_features)]

        return [features_sacc, features_fix]

    def train(self, labeled_feature_values):

        feature_values, feature_labels = self.separate_feature_labels(labeled_feature_values)

        # if top features are given it will reduce features
        feature_values = self.select_top_features(feature_values)

        print("Training")
        print("after top features", feature_values[0].shape, feature_values[1].shape)
        start_time = datetime.datetime.now()
        self.clf_sac.fit(feature_values[0], feature_labels[0])
        self.clf_fix.fit(feature_values[1], feature_labels[1])
        print("train time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))
