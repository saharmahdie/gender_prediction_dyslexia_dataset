import datetime
import os


from schau_mir_in_die_augen.evaluation.base_evaluation import BaseEvaluation


class OurEvaluationOne(BaseEvaluation):

    def __init__(self, base_clf, vel_threshold=50, min_fix_duration=.1):
        super().__init__()
        self.min_fix_duration = min_fix_duration
        self.vel_threshold = vel_threshold
        self.clf = base_clf
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

    def trajectory_split_and_feature(self, trajectory):
        """ Generate feature vectors for all saccades and fixations in a trajectory

        :param trajectory: ndarray
            2D array of gaze points (x,y)
        :return: list, list
            list of feature vectors (each is a 1D ndarray)
        """

        # get feature dataFrame
        features = self.trajectory_split_and_feature_basic(trajectory)

        # filter features we use
        # features.drop(['sample_type'], axis=1, inplace=True)

        return [features]

    def train(self, labeled_feature_values):

        feature_values, feature_labels = self.separate_feature_labels(labeled_feature_values)

        self.feature = feature_values[0].columns.array

        # if top features are given it will reduce features
        feature_values = self.select_top_features(feature_values)

        print("Training")
        start_time = datetime.datetime.now()
        self.clf.fit(feature_values[0], feature_labels[0])
        print("train time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))
