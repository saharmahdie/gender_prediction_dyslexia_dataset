import unittest
import numpy as np

import schau_mir_in_die_augen.features as feat
import schau_mir_in_die_augen.process.trajectory
import schau_mir_in_die_augen.process.conversion as converters
from schau_mir_in_die_augen.features import BasicFeatures1D
from schau_mir_in_die_augen.process.trajectory import calculate_derivation, Trajectory
from schau_mir_in_die_augen.feature_extraction import all_features, sac_subset, fix_subset


class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.xy = np.random.randint(-100, 1500, (30, 2))
        self.screen_params = {'pix_per_mm': np.asarray([3.5443037974683542, 3.5443037974683542]),
                              'screen_dist_mm': 550.,
                              'screen_res': np.asarray([1680, 1050])}
        self.trajectory = Trajectory(self.xy, 'pixel', sample_rate=1,
                                     **Trajectory.screen_params_converter(self.screen_params))

    def test_stat_names(self):
        v = np.asarray([1, 2, 3])
        self.assertEqual(len(feat.statistics(v)), len(feat.stat_names(extended=True)))

    def test_statistics(self):
        v = [1, 2, 3]
        self.assertAlmostEqual(feat.statistics(np.asarray(v)),
                               [2.0, 2.0, 3.0, 0.816496580927726, 0.0, -1.5, 1.0, 0.6666666666666666])

        # empty input
        self.assertEqual(feat.statistics(np.asarray([])), [0, 0, 0, 0, 0, 0, 0, 0])

    def test_compare_statistics(self):
        v = np.random.rand(1000)

        feature_names = [BasicFeatures1D.mean,
                         BasicFeatures1D.median,
                         BasicFeatures1D.max,
                         BasicFeatures1D.std,
                         BasicFeatures1D.skew,
                         BasicFeatures1D.kurtosis,
                         BasicFeatures1D.min,
                         BasicFeatures1D.var]

        features1 = dict(zip(feat.stat_names('', True), feat.statistics(v)))
        features2 = feat.basic_features_1d(v, feature_names)
        # noinspection PyTypeChecker
        for name in feature_names:
            self.assertAlmostEqual(features1['_'+name.name],features2[name])

    def test_calculate_dispersion(self):
        x = [0, 2, 3]
        y = [1, 2, 3]
        self.assertEqual(5, feat.calculate_dispersion(np.asarray([x, y]).T))

    def test_angle_between_first_and_last_points(self):
        x = [0, 2, 1]
        y = [0, 2, 1]
        a = feat.angle_between_first_and_last_points(np.asarray([x, y]).T)
        self.assertAlmostEqual(a, 45)

        x = [0, 2, 3]
        y = [0, 2, 3]
        self.assertAlmostEqual(feat.angle_between_first_and_last_points(np.asarray([x, y]).T), 45)

        # empty input
        # self.assertEqual(feat.angle_between_first_and_last_points(np.asarray([])), 0)

    def test_pixelconvert_gaze_pixel_into_eye_coordinates_angle_convert(self):
        screen_size_mm = np.asarray([474.0, 297.0])
        screen_res = np.asarray([1680.0, 1050.0])
        pix_per_mm = screen_res / screen_size_mm
        screen_dist_mm = 550.0
        x1 = [0., 0.]
        y1 = [1050. / 2, 1050. / 2]
        xy_pix = np.asarray([x1, y1]).T
        res = [[-237., 0., 550.], [-237., 0., 550.]]
        b = converters.convert_gaze_pixel_into_eye_coordinates(xy_pix, pix_per_mm, screen_dist_mm, screen_res)
        self.assertTrue(np.allclose(b, res))

    def test_angle_with_previous_fix(self):
        p1 = np.array([[237., 0., 550.], [237., 0., 550.]])
        p2 = np.array([[-237., 0., 550.], [-237., 0., 550.]])
        self.assertAlmostEqual(feat.angle_with_previous_fix(p1, p2), 46.62329869725378)

    # def test_angle_with_previous_fix(self):
    #     x1 = [-1, 1]
    #     y1 = [-1, 1]
    #     x2 = [0, 2]
    #     y2 = [0, 2]
    #     xy1 = np.asarray([x1, y1]).T
    #     xy2 = np.asarray([x2, y2]).T
    #     a = feat.angle_with_previous_fix(xy1, xy2)
    #     self.assertAlmostEqual(a, 45)

        # x = [0, 0, 0]
        # y = [0, 0, 0]
        # prev_x = [2, 3, 4]
        # prev_y = [2, 3, 4]
        # self.assertAlmostEqual(feat.angle_with_previous_fix(np.asarray([x, y]).T, np.asarray([prev_x, prev_y]).T), 45)
        #
        # # empty input
        # # self.assertEqual(feat.angle_with_previous_fix(np.asarray([])), 0)

    def test_angle_btw_2consecutive_points_in_vector(self):
        a = feat.angle_btw_2consecutive_points_in_vector(self.xy)
        self.assertEqual(len(a), len(self.xy) - 1)

        x = [0, 2, 3]
        y = [0, 2, 3]
        res = feat.angle_btw_2consecutive_points_in_vector(np.asarray([x, y]).T)
        self.assertTrue(np.allclose(res, [45., 45.]))

        # empty input
        # self.assertEqual(feat.angle_btw_2consecutive_points_in_vector(np.asarray([])), 0)

    # def test_angular_velocity(self):
    #     x = np.asarray([0, 45, 80, 70])
    #     res = feat.angular_velocity(x, 250)
    #     exp = 250 * np.asarray([45, 35, 10])
    #     self.assertTrue(np.allclose(res, exp))

    def test_angle_among_3consecutive_Points(self):
        x = [0, 2, 3, 2, 3]
        y = [0, 2, 3, 2, 1]
        p = np.asarray([x, y]).T
        res = feat.angle_among_3consecutive_Points(p)
        self.assertTrue(np.allclose(res, [0, 180, 90]))

    def test_angular_acceleration(self):
        x = np.asarray([0, 45, 80, -70])
        res = schau_mir_in_die_augen.process.trajectory.calculate_derivation(x, 250)
        exp = 250 * np.asarray([45, 35, -150])
        self.assertTrue(np.allclose(res, exp))

    def test_calculate_velocity(self):
        x = np.asarray([0, 45, 80, 70])
        y = np.asarray([0, 45, 80, 70])
        xy = np.asarray([x, y]).T
        res = calculate_derivation(xy, 250)
        exp_a = 250 * np.asarray([45, 35, -10])
        exp = np.asarray([exp_a, exp_a]).T
        self.assertTrue(np.allclose(res, exp))

    def test_calculate_distance_vector(self):
        x = [0, 0, 1, 2, 0, 3]
        y = [0, 0, 0, 0, 0, 3]
        p = np.asarray([x, y]).T
        res = feat.calculate_distance_vector(p)
        self.assertTrue(np.allclose(res, [0, 1, 1, np.sqrt(4), np.sqrt(18)]))

    def test_total_length(self):
        v = np.asarray([0, 1, 2])
        res = feat.total_length(v)
        self.assertAlmostEqual(res, 3)

    def test_distance_between_first_and_last_points(self):
        x = [0, 2, 3, 2, 1]
        y = [0, 2, 3, 2, 0]
        p = np.asarray([x, y]).T
        res = feat.distance_between_first_and_last_points(p)
        self.assertAlmostEqual(res, 1)

    def test_distance_from_previous_fix_or_sacc(self):
        x1 = [-1, 1]
        y1 = [-1, 1]
        x2 = [0, 2]
        y2 = [0, 0]
        xy1 = np.asarray([x1, y1]).T
        xy2 = np.asarray([x2, y2]).T
        a = feat.distance_cog(xy1, xy2)
        self.assertAlmostEqual(a, 1)

        x = [0, 0, 0]
        y = [0, 0, 0]
        prev_x = [2, 3, 4]
        prev_y = [2, 3, 4]
        res = feat.distance_cog(np.asarray([x, y]).T, np.asarray([prev_x, prev_y]).T)
        self.assertAlmostEqual(res, np.sqrt(9+9))

    def test_saccade_feature(self):
        # we only want a 1D array from the feature extraction code
        all_feats = all_features(self.trajectory)
        sac_feats = all_feats[sac_subset()]
        self.assertEqual(sac_feats.shape[0], 1)
        self.assertGreater(sac_feats.shape[1], 0, "The feature vector should not be empty.")

    def test_fixation_feature(self):
        # we only want a 1D array from the feature extraction code
        all_feats = all_features(self.trajectory)
        fix_feats = all_feats[fix_subset()]
        self.assertEqual(fix_feats.shape[0], 1)
        self.assertGreater(fix_feats.shape[1], 0, "The feature vector should not be empty.")

    def test_general_gaze_points_features_names(self):
        f = feat.general_gaze_points_features(self.xy, 250, self.screen_params)
        self.assertEqual(len(f[0]), len(f[1]))

    def test_general_gaze_points_features(self):
        # we only want a 1D array from the feature extraction code
        res = feat.general_gaze_points_features(self.xy, 250, self.screen_params)[0]
        self.assertEqual(len(np.asarray(res).shape), 1)
        self.assertGreater(len(res), 0, "The feature vector should contain not be empty.")

    def test_non_distributional_fixation_features(self):
        pass

    def test_non_distributional_saccade_features(self):
        pass

    def test_ngram_bins(self):
        angle_2points_list = np.array([10, 30, 50, -10, -80, -175, -80, -175])
        binned_data_result = [5, 6, 6, 5, 3, 1, 3, 1]
        binned_data = feat.ngram_bins(angle_2points_list, 8)
        self.assertTrue(np.allclose(binned_data, binned_data_result))

    def test_ngram_features(self):
        angle_2points_list = np.array([10, 30, 50, -10, -80, -175, -80, -175])
        binned_data = feat.ngram_bins(angle_2points_list, 8)

        res_unigram = [2.,  0.,  2.,  0.,  2.,  2.,  0.,  0.,  1.,  1.,  2.,  1.,  0., -2.,  0.,  1.,  8.]
        res_bigram = np.concatenate((np.array([0, 0, 1]), np.zeros(13), (np.array([2])), np.zeros(17), (np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1])), np.zeros(18), (np.array([0.109375, 0.0, 2, 0.35869500885153116, 3.4582389089405248, 12.037036103539782, 0, 0.128662109375, 7.0]))), axis=0)

        unigram = feat.ngram_features(binned_data, 1)
        bigram = feat.ngram_features(binned_data, 2)

        self.assertTrue(np.allclose(unigram[0], res_unigram))
        self.assertTrue(np.allclose(bigram[0], res_bigram))

    def test_histogramgram_features(self):
        angle_3points = np.asarray([1, 1, 1, 2, 2, 360, 360, 360, 360])
        results = np.concatenate((np.array([0, 3, 2]), np.zeros(356), np.array([4, 0.025, 0.00, 4.00, 0.28271992422812287, 11.903268085712865, 146.22089959171234, 0.00, 0.07993055555555555, 9.00])), axis=0)
        p = np.asarray([angle_3points]).T
        res = feat.histogram(p, 360)
        self.assertTrue(np.allclose(res[0], results))

    def test_micro_fixation(self):
        x = [0, 0, 1, 2, 0, 3, 1, 1]
        y = [0, 0, 0, 0, 0, 3, 1, 1]
        p = np.asarray([x, y]).T

        results = np.concatenate((np.array([6, 2, 4, 2]), np.zeros(19), np.array([0.60869565, 0, 6, 1.49605657, 2.55548441, 5.56825237 , 0, 2.23818526])), axis=0)
        results_with_dis_threshold = np.concatenate((np.array([4, 1, 3, 0, 0, 0, 1]), np.zeros(16), np.array([0.39130435, 0, 4, 1.01034348, 2.70056913, 6.03855967, 0, 1.02079395])), axis=0)

        res_without_dis_threshold = feat.micro_fixation(p)
        res_with_dis_threshold    = feat.micro_fixation(p,1)
        self.assertTrue(np.allclose(res_without_dis_threshold[0], results))
        self.assertTrue(np.allclose(res_with_dis_threshold[0], results_with_dis_threshold))