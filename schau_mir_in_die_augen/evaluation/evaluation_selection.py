
import datetime
import numpy as np

from schau_mir_in_die_augen.evaluation.evaluation_general import EvaluationGeneralFixSac


class EvaluationSelection(EvaluationGeneralFixSac):
    def __init__(self, base_clf, vel_threshold=50, min_fix_duration=.1, paper_only=False, feature_number=0, feature_list=""):
        """

        :param base_clf:
        :param vel_threshold:
        :param min_fix_duration:
        :param paper_only:
        :param feature_number: HardCode Select of Features
        :param feature_list: select feature list according to the classification task ("dyslexia_prediction_features", "healthy_gender_prediction_features", "dyslexic_gender_prediction_features")
        """

        super().__init__(base_clf=base_clf, min_fix_duration=min_fix_duration, vel_threshold=vel_threshold)

        self.paper_only = paper_only
        self.feature_number = feature_number
        self.feature_list = feature_list

    def trajectory_split_and_feature(self, trajectory):
        """ Generate feature vectors for all saccades and fixations in a trajectory

        :param trajectory: ndarray
            2D array of gaze points (x,y)
        :return: list, list
            list of feature vectors (each is a 1D ndarray)
        """

        # get feature dataFrame
        features = self.trajectory_split_and_feature_basic(trajectory=trajectory)

        features_sacc = features[(features['sample_type'] == 'saccade') & features['duration'] > 0.012]
        features_fix = features[features['sample_type'] == 'fixation']
        features_sacc = features_sacc.drop(['sample_type'], axis=1)
        features_fix = features_fix.drop(['sample_type'], axis=1)
        fix_dur = []
        sac_dur = []


        ### the new features
        features_sacc['num_sacc'] = len(features_sacc)
        sac_dur.append(features_sacc['duration'])
        features_sacc['sac_dur_mean'] = np.mean(sac_dur)
        features_sacc['sac_dur_tot'] = np.sum(sac_dur)

        ### the new features
        fix_dur.append(features_fix['duration'])
        features_fix['fix_dur_mean'] = np.mean(fix_dur)
        features_fix['fix_dur_tot'] = np.sum(fix_dur)
        #####

        if features_sacc['sac_dur_tot'].iloc[0] > 0:
            features_fix['RFDSD'] = np.abs(np.divide(features_fix['fix_dur_tot'].iloc[0], features_sacc['sac_dur_tot'].iloc[0]))
        else:
            features_fix['RFDSD'] = 0


        ## select Features
        ## top Feature lists (fisher score calculated by using the gender of 88 healthy participants features) to predict healthy gender (we used the top 4)

        if self.feature_list == "healthy_gender_prediction_features":
            features_fix_list = ['avg_vel', 'ang_vel_mean', 'win_amplitude', 'dispersion', 'ang_vel_median', 'std_y', 'ang_vel_var', 'RFDSD', 'win_ratio' ,'ang_vel_std' ,'ang_acc_median' ,'vel_x_skew']
            features_sacc_list = ['skew_x', 'skew_y', 'acc_y_std', 'acc_y_max', 'vel_y_std', 'vel_y_var', 'vel_y_max', 'acc_y_var', 'std_y', 'ang_vel_std', 'vel_y_skew', 'ang_vel_max']

        ## top Feature lists (fisher score calculated by using the gender of  97 dyslexic participants features) to predict disabled gender (we used the top 4)
        elif self.feature_list == "dyslexic_gender_prediction_features":
            features_fix_list = ['win_ratio', 'ang_vel_std', 'ang_vel_var', 'ang_vel_max', 'avg_vel', 'ang_vel_mean', 'ang_vel_min', 'vel_y_var', 'win_amplitude', 'dispersion', 'fix_dur_mean', 'RFDSD']
            features_sacc_list = ['num_sacc', 'ang_vel_median', 'sac_dur_tot', 'vel_x_max', 'acc_x_max', 'acc_x_std', 'acc_x_var', 'ang_vel_mean', 'ang_acc_skew', 'avg_vel', 'skew_y', 'ang_vel_skew']

        # ## top Feature lists (fisher score calculated by using the dyslexia of all 185 mixed participants features) to predict dyslexia
        elif self.feature_list == "dyslexia_prediction_features":
            features_fix_list = ['fix_dur_tot', 'dist_prev_win', 'RFDSD', 'win_ratio', 'fix_dur_mean', 'ang_acc_mean', 'angular_vel_total', 'ang_acc_median', 'path_len', 'ang_acc_max', 'ang_acc_skew', 'win_angle']
            features_sacc_list = ['sac_dur_mean', 'ang_acc_max', 'dispersion', 'angular_vel_total', 'ang_vel_max', 'std_x', 'acc_y_kurtosis', 'vel_y_kurtosis', 'path_len', 'vel_x_kurtosis', 'ang_acc_min', 'vel_x_skew']
        
        # ## top Feature lists (ANOVA calculated by using the dyslexia of all 185 mixed participants features) to predict dyslexia
        elif self.feature_list == "dyslexia_prediction_ANOVA_features":
            features_fix_list = ['fix_dur_tot', 'dist_prev_win', 'acc_total', 'RFDSD', 'win_ratio', 'fix_dur_mean', 'ang_acc_mean', 'angular_vel_total', 'ang_acc_median', 'path_len', 'ang_acc_max', 'ang_acc_skew']
            features_sacc_list = ['sac_dur_mean', 'angular_vel_total', 'dispersion', 'ang_acc_max', 'ang_vel_max', 'std_x', 'vel_y_kurtosis', 'acc_y_kurtosis', 'path_len', 'vel_x_kurtosis', 'ang_acc_min', 'vel_x_skew']

        # ## top Feature lists (CHI calculated by using the dyslexia of all 185 mixed participants features) to predict dyslexia
        elif self.feature_list == "dyslexia_prediction_Chi_features":
            features_fix_list = ['fix_dur_tot', 'dist_prev_win', 'RFDSD', 'acc_total', 'fix_dur_mean', 'win_ratio', 'angular_vel_total', 'ang_acc_mean', 'path_len', 'ang_acc_median', 'ang_vel_min', 'ang_acc_max']
            features_sacc_list = ['acc_total', 'sac_dur_mean', 'angular_vel_total', 'path_len', 'ang_acc_max', 'dispersion', 'std_x', 'ang_vel_max', 'vel_x_kurtosis', 'std_y', 'win_amplitude', 'acc_x_kurtosis']

        elif self.feature_list == "prediction_of_healthy_gender_group_chi_method":
            features_fix_list = ['RFDSD','ang_vel_mean','ang_vel_kurtosis','avg_vel','ang_vel_median','win_amplitude','dispersion','fix_dur_mean','win_ratio','std_y','ang_vel_skew','path_len']
            features_sacc_list = ['skew_x','vel_y_var','acc_y_var','acc_y_max','acc_y_std','vel_y_max','vel_y_std','skew_y','std_y','vel_x_kurtosis','ang_vel_max','acc_x_max']


        elif self.feature_list == "prediction_of_healthy_gender_group_rfecv_method":
            features_fix_list = ['acc_x_skew','acc_x_std','acc_x_var','acc_y_kurtosis','acc_y_max','acc_y_mean','acc_y_median','acc_y_min','acc_y_skew','acc_y_std','acc_y_var','ang_acc_kurtosis']
            features_sacc_list = ['dispersion','kurtosis_x','kurtosis_y','num_sacc','sac_dur_mean','skew_x','std_y','vel_x_kurtosis','vel_x_max','vel_x_mean','vel_x_median','vel_x_skew','vel_x_std']


        elif self.feature_list == "prediction_of_healthy_gender_group_ANOVA_method":
            features_fix_list = ['RFDSD','avg_vel','ang_vel_mean','ang_vel_median','win_ratio','ang_vel_std','dispersion','fix_dur_mean','ang_vel_var','win_amplitude','ang_vel_kurtosis','std_y']
            features_sacc_list = ['skew_x','skew_y','acc_y_std','acc_y_max','vel_y_std','vel_y_max','vel_y_skew','ang_vel_max','ang_vel_std','ang_acc_min','vel_x_kurtosis','acc_x_max']

        elif self.feature_list == "prediction_of_healthy_gender_group_ANOVA_method_after_correlation":
            features_fix_list = ['RFDSD', 'avg_vel', 'win_ratio', 'ang_vel_kurtosis']
            features_sacc_list = ['skew_x', 'skew_y', 'acc_y_std', 'vel_y_skew']


        elif self.feature_list == "prediction_of_healthy_gender_and_dyslexic_gender":
        # top 8 Feature to predict healthy gender + top 8 Feature to predict dyslexic gender lists
            features_fix_list = ['win_ratio', 'avg_vel', 'ang_vel_std','ang_vel_mean' , 'ang_vel_var' , 'win_amplitude', 'ang_vel_max', 'dispersion']
            features_sacc_list = ['num_sacc', 'skew_x', 'ang_vel_median', 'skew_y', 'sac_dur_tot', 'acc_y_std', 'vel_x_max', 'acc_y_max']

        elif self.feature_list == "mixed_predict_gender_and_predict_dyslexia":
        # top 8 Feature mixed predict gender + 8 top Features predict dyslexia
            features_fix_list = ['fix_dur_mean', 'fix_dur_tot', 'RFDSD', 'dist_prev_win', 'acc_x_median', 'win_ratio', 'acc_x_mean', 'ang_acc_mean']
            features_sacc_list = ['angle_prev_win', 'sac_dur_mean', 'win_amplitude', 'ang_acc_max', 'std_x', 'dispersion', 'vel_y_std', 'angular_vel_total']


        ### top Feature lists (fisher score calculated by using the gender of all the 185 mixed participants features) to predict gender (we used the top 4)
        elif self.feature_list == "mixed_predict_gender":
            features_fix_list = ['fix_dur_mean', 'RFDSD', 'fix_dur_tot', 'acc_x_median', 'acc_x_mean', 'acc_y_min',
                                 'vel_x_skew', 'ang_vel_mean', 'acc_y_var', 'acc_y_skew', 'ang_vel_min', 'vel_x_min',
                                 'acc_y_std', 'spatial_density', 'ang_vel_skew', 'ang_acc_median', 'win_amplitude',
                                 'std_x', 'win_ratio', 'avg_vel']
            features_sacc_list = ['angle_prev_win', 'win_amplitude', 'std_x', 'vel_y_std', 'acc_y_std', 'vel_x_median',
                                  'vel_y_max', 'acc_y_min', 'vel_x_std', 'ang_acc_median', 'vel_y_min', 'acc_y_max',
                                  'vel_x_mean', 'skew_x', 'vel_y_var', 'std_y','acc_y_var', 'vel_x_min', 'vel_x_var',
                                  'acc_x_median']

        else:
            features_fix_list = []
            features_sacc_list = []
            print("There is no features:: Please select feature list")



        if self.feature_number % 2 != 0:
            raise Exception("Feature Number has to be even, got ", self.feature_number)

        # reduce by feature list variable
        features_sacc = features_sacc[features_sacc_list[:self.feature_number // 2]]
        features_fix = features_fix[features_fix_list[:self.feature_number // 2]]


        return [features_sacc, features_fix]

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

    def evaluation(self, feature_values):
        return self.weighted_evaluation(labeled_feature_values=feature_values, weights=[0.5, 0.5])  ## [sac, fix]
