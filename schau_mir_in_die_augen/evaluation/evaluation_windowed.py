import pandas as pd
import numpy as np
import datetime

from schau_mir_in_die_augen.feature_extraction import all_features
from schau_mir_in_die_augen.evaluation.base_evaluation import BaseEvaluation
from schau_mir_in_die_augen.trajectory_classification.trajectory_split import sliding_window


class EvaluationWindowed(BaseEvaluation):
    def __init__(self, base_clf, window_size=100):
        super().__init__()
        self.window_size = window_size
        self.clf = base_clf

    def get_split_parameter(self):
        """ Return specific Parameter for split
        e.g. IVT with velocity threshold 50 deg/s and minimal fixation duration of 0.1 seconds
        """
        split_method = 'windowed'
        split_parameter = {}
        clf_parameter = {}

        return split_method, split_parameter, clf_parameter

    def trajectory_split_and_feature(self, trajectory):
        """ Generate feature vectors for all saccades and fixations
        and our saccade, fixation, general features in a trajectory
        :param trajectory: Trajectory
            2D array of gaze points (x,y) and more
        :return: list, list
            list of feature vectors (each is a 1D ndarray)
        """

        trajectory.convert_to('angle')
        trajectory.apply_filter('savgol')

        window_size = self.window_size
        windows = sliding_window(len(trajectory), window_size, window_size//2)

        features = pd.DataFrame()
        prev_part = np.asarray([])

        for i, s in enumerate(windows):
            # slice expects start and stop, where stop is exclusive
            part = slice(s[0], s[1] + 1)
            all_feats = all_features(trajectory, part, prev_part=prev_part, omit_stats=False, omit_our=False)
            features = features.append(all_feats)
            prev_part = part

        return [features]

    def train(self, labeled_feature_values):

        feature_values, feature_labels = self.separate_feature_labels(labeled_feature_values)

        # if top features are given it will reduce features
        feature_values = self.select_top_features(feature_values)

        print("Training")
        start_time = datetime.datetime.now()
        self.clf.fit(feature_values[0], feature_labels[0])
        print("train time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))
