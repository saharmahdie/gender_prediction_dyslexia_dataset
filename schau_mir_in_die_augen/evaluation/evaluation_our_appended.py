from schau_mir_in_die_augen.evaluation.evaluation_general import EvaluationGeneralFixSac
from schau_mir_in_die_augen.feature_extraction import paper_all_subset


class OurEvaluationAppended(EvaluationGeneralFixSac):

    def __init__(self, base_clf,
                 vel_threshold=50, min_fix_duration=.1,
                 paper_only=False):

        super().__init__(base_clf=base_clf, min_fix_duration=min_fix_duration, vel_threshold=vel_threshold)

        self.paper_only = paper_only

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
        features_sacc = features[(features['sample_type'] == 'saccade') & features['data_data_duration'] > 0.012]
        features_fix = features[features['sample_type'] == 'fixation']
        # remove 'is_saccade' column
        features_sacc.drop(['sample_type'], axis=1, inplace=True)
        features_fix.drop(['sample_type'], axis=1, inplace=True)

        # reduce dataFrame,
        if self.paper_only:
            features_fix = features_fix[paper_all_subset()]
            features_sacc = features_sacc[paper_all_subset()]

        return [features_sacc, features_fix]
