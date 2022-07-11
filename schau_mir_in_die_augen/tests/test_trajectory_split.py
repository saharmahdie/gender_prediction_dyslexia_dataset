import numpy as np
from unittest import TestCase

from schau_mir_in_die_augen.trajectory_classification.trajectory_split import set_sample_type, SetSampleError, \
    mark_nan_sample, EyeMovementClassification, EyeMovementClassifier
from schau_mir_in_die_augen.process.trajectory import Trajectory


class TestIvt(TestCase):

    def setUp(self):
        self.isnan = [False] * 20
        self.sample_start = [0,1,10,15,16,17,19,20]
        self.sample_type = ['a1','a2','b','c1','c2','c3','d']

        # 1, 1, 2, 2, 2, 2, Nan, Nan, 0, 0, 5, 0
        self.xy = [(0,0),(1,-1),(2,-2),(3,0),(4,2),(5,4),(6,6),(np.nan,np.nan),(8,8),(8,8),(8,8),(4,4),(4,4)]
        self.trajectory = Trajectory(self.xy,'angle_deg')
        self.subset_classifier = EyeMovementClassifier('IVT',
                                                       {'vel_threshold': 1.5,
                                                        'min_fix_duration': 0,
                                                        'sample_rate': 1})

    def test_set_sample_single(self):

        new = 'new'

        # single element or negative range
        self.assertRaises(SetSampleError, set_sample_type, (5, 5), new, self.sample_start, self.sample_type)
        self.assertRaises(SetSampleError, set_sample_type, (5, 3), new, self.sample_start, self.sample_type)

        # before
        sample_start, sample_type = set_sample_type((0, 1), new, self.sample_start[1:], self.sample_type[1:])
        self.assertEqual([0, 1, 10, 15, 16, 17, 19, 20], sample_start)
        self.assertEqual([new, 'a2', 'b', 'c1', 'c2', 'c3', 'd'], sample_type)

        # first
        sample_start, sample_type = set_sample_type((0, 1), new, self.sample_start, self.sample_type)
        self.assertEqual([0, 1, 10, 15, 16, 17, 19, 20], sample_start)
        self.assertEqual([new, 'a2', 'b', 'c1', 'c2', 'c3', 'd'], sample_type)

        # between
        sample_start, sample_type = set_sample_type((16, 17), new, self.sample_start, self.sample_type)
        self.assertEqual([0, 1, 10, 15, 16, 17, 19, 20], sample_start)
        self.assertEqual(['a1', 'a2', 'b', 'c1', new, 'c3', 'd'], sample_type)

        # in between
        sample_start, sample_type = set_sample_type((4, 7), new, self.sample_start, self.sample_type)
        self.assertEqual([0, 1, 4, 7, 10, 15, 16, 17, 19, 20], sample_start)
        self.assertEqual(['a1', 'a2', new, 'a2', 'b', 'c1', 'c2', 'c3', 'd'], sample_type)

        # last
        sample_start, sample_type = set_sample_type((19, 20), new, self.sample_start, self.sample_type)
        self.assertEqual([0, 1, 10, 15, 16, 17, 19, 20], sample_start)
        self.assertEqual(['a1', 'a2', 'b', 'c1', 'c2', 'c3', new], sample_type)

        # after
        self.assertRaises(SetSampleError, set_sample_type, (20, 21), new, self.sample_start, self.sample_type)
        self.assertRaises(SetSampleError, set_sample_type, (21, 22), new, self.sample_start, self.sample_type)

    def test_mark_nan_sample(self):

        # Trajectory:   0   1   2   3   4   5
        # Velocity:       0   1   2   3   4

        isnan = self.isnan

        # none
        sample_start, sample_type = mark_nan_sample(isnan, self.sample_start, self.sample_type)
        self.assertEqual([0, 1, 10, 15, 16, 17, 19, 20], sample_start)
        self.assertEqual(['a1', 'a2', 'b', 'c1', 'c2', 'c3', 'd'], sample_type)

        # single
        isnan[12] = True
        isnan[16] = True

        sample_start, sample_type = mark_nan_sample(isnan, self.sample_start, self.sample_type)
        self.assertEqual([0, 1, 10, 11, 13, 15, 17, 19, 20], sample_start)
        self.assertEqual(['a1', 'a2', 'b', 'NaN', 'b', 'NaN', 'c3', 'd'], sample_type)

        # first, region, last
        isnan[0] = True
        isnan[13] = True
        isnan[14] = True
        isnan[15] = True
        isnan[19] = True
        sample_start, sample_type = mark_nan_sample(isnan, self.sample_start, self.sample_type)
        self.assertEqual([0, 1, 10, 11, 17, 18, 20], sample_start)
        self.assertEqual(['NaN', 'a2', 'b', 'NaN', 'c3', 'NaN'], sample_type)

        # first region, region, last region
        isnan[1] = True
        isnan[18] = True
        sample_start, sample_type = mark_nan_sample(isnan, self.sample_start, self.sample_type)
        self.assertEqual([0, 2, 10, 11, 17, 20], sample_start)
        self.assertEqual(['NaN', 'a2', 'b', 'NaN', 'NaN'], sample_type)

        # all
        for ii in range(2, 12):
            isnan[ii] = True
        isnan[17] = True
        sample_start, sample_type = mark_nan_sample(isnan, self.sample_start, self.sample_type)
        self.assertEqual([0, 20], sample_start)
        self.assertEqual(['NaN'], sample_type)

    def test_emc(self):

        self.assertTrue(isinstance(self.subset_classifier,EyeMovementClassifier))

        emc = EyeMovementClassification(self.trajectory, self.subset_classifier)
        emc._subset_classifier[0].minimum_sample_length = 1  # the default is now 5
        emc.do_classification()
        sample_start, sample_type = emc.get_subs(0,0)

        self.assertEqual([0, 2, 8, 10, 11, 12], sample_start)
        self.assertEqual(['fixation', 'saccade', 'fixation', 'saccade', 'fixation'], sample_type)

        emc._subset_classifier[0].minimum_sample_length = 2
        emc.do_classification()  # todo: this should not be necessary. changes on clf, should trigger something
        sample_start, sample_type = emc.get_subs([0], [0])

        self.assertEqual([0, 2, 8, 10, 11, 12], sample_start[0][0])
        self.assertEqual(['fixation', 'saccade', 'fixation', 'short saccade', 'short fixation'], sample_type[0][0])

        emc._subset_classifier[0].mark_nan = True
        sample_start, sample_type = emc.do_classification()

        self.assertEqual([0, 2, 6, 8, 10, 11, 12], sample_start[0][0])
        self.assertEqual(['fixation', 'saccade', 'NaN', 'fixation', 'short saccade', 'short fixation'], sample_type[0][0])



