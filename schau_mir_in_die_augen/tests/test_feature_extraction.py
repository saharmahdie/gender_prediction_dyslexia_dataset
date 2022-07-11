import unittest
import numpy as np
import pandas as pd

from schau_mir_in_die_augen.feature_extraction import all_features, all_features2, \
    all_subset, fix_subset, sac_subset, \
    translate_feature_list, get_features
from schau_mir_in_die_augen.process.trajectory import Trajectory
from schau_mir_in_die_augen.features import BasicFeatures1D, BasicFeatures2D, ExtendedFeatures2D, \
    TimeFeatures, HistoryFeatures
from schau_mir_in_die_augen.feature_extraction import Data, Dimension, ComplexFeatures


class TestSaccades(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.xy = np.random.randint(-100, 1500, (30, 2))
        self.screen_params = {'pix_per_mm': np.asarray([3.5443037974683542, 3.5443037974683542]),
                              'screen_dist_mm': 550.,
                              'screen_res': np.asarray([1680, 1050])}
        self.trajectory = Trajectory(self.xy, 'pixel', sample_rate=1,
                                     **Trajectory.screen_params_converter(self.screen_params))

    def test_statistics(self):
        all_feats = all_features(self.trajectory)
        # print(all_feats.columns)
        self.assertEqual(len(all_feats[fix_subset(True)].columns), 9)
        self.assertEqual(len(all_feats[fix_subset(False)].columns), 9)

        self.assertEqual(len(all_feats[sac_subset(True)].columns), 43)
        self.assertEqual(len(all_feats[sac_subset(False)].columns), 40)

    def test_feature_length(self):

        feature_list1 = all_subset(omit_our=False, omit_stats=False)
        self.assertEqual(320, len(feature_list1))

        feature_list2 = all_subset(omit_our=False, omit_stats=True)
        self.assertEqual(272, len(feature_list2))

        feature_list3 = all_subset(omit_our=True, omit_stats=False)
        self.assertEqual(64, len(feature_list3))
        feature_list3_new = translate_feature_list(feature_list3)
        self.assertEqual(63, len(feature_list3_new))

        feature_list4 = all_subset(omit_our=True, omit_stats=True)
        self.assertEqual(16, len(feature_list4))

        feature_list = all_subset()
        self.assertEqual(64, len(feature_list))

    def test_feature(self):

        all_feats1 = all_features(self.trajectory)
        all_feats2 = pd.DataFrame(all_features2(self.trajectory), index=[0])
        self.assertEqual(len(list(all_feats1)), len(list(all_feats2))+1)  # spatial density

        self.assertEqual(1, all_feats2.shape[0])

        for feat1 in all_feats1:
            match = [feat2 for feat2 in all_feats2 if all_feats1.iloc[0][feat1] == all_feats2.iloc[0][feat2]]
            if len(match):
                all_feats2.drop(columns=match[0], inplace=True)
                if len(match) > 1:
                    print('Deleted "{feat1}" with value "{value} for "{feat2}" with the same value.'
                          'Could be a mistake: Value is ambiguous!'.format(feat1=feat1,feat2=match[0],
                                                                           value=all_feats1.iloc[0][feat1]))
            else:
                print('No Match for {}'.format(feat1))

        self.assertEqual(0, len(list(all_feats2)))

    def test_get_features(self):
        """ This is an example how to use it """

        # feature_entries shoul be set, so no feature accoure double

        feature_entries = {(Data.data, Dimension.data, TimeFeatures.duration),
                           (Data.pixel_data, Dimension.data, BasicFeatures1D.len),
                           (Data.pixel_difference, Dimension.x_data, BasicFeatures1D.mean),
                           (Data.deg_velocity, Dimension.xy_length, BasicFeatures1D.max),
                           (Data.deg_velocity_change, Dimension.data, BasicFeatures1D.sum),
                           (Data.deg_acceleration, Dimension.xy_length, BasicFeatures1D.sum),
                           (Data.deg_data, Dimension.xy_data, BasicFeatures2D.dispersion),
                           (Data.deg_data, Dimension.xy_data, ExtendedFeatures2D.angle_first_last)}

        # There are features which need a previous data set
        feature_entries.add((Data.pixel_data, Dimension.xy_data, HistoryFeatures.distance_cog))

        # There are some complex features which need other features. Make Shure, needed Features are loaded.
        # todo: this should be done automatically
        feature_entries.add((Data.pixel_difference, Dimension.xy_length, BasicFeatures1D.sum))
        feature_entries.add((Data.data, Dimension.data, TimeFeatures.duration))  # duplicate is no problem
        feature_entries.add((Data.complex, ComplexFeatures.avg_vel))
        # Feature called "avg_vel" = path_len / duration
        # Old name conventions is confusing, would expect:
        feature_entries.add((Data.pixel_data, Dimension.xy_length, BasicFeatures1D.mean))

        # here the features are calculated
        features = get_features(self.trajectory, feature_entries,
                                part=slice(len(self.trajectory)//2, None),
                                prev_part=slice(0, len(self.trajectory)//2))
        self.assertEqual(12, len(features))
