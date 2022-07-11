from unittest import TestCase
import numpy as np

import schau_mir_in_die_augen.features as feat
from schau_mir_in_die_augen.trajectory_classification.trajectory_split import ivt, eye_movement_classification,\
    EyeMovementClassifier
from schau_mir_in_die_augen.process.trajectory import Trajectory

import time


class TestIvt(TestCase):
    def setUp(self):
        #    0  1  2  3    4    5
        x = [0, 0, 0, 0, 0.1, 0.2]
        y = [0, 0, 0, 0, 0.1, 0.2]
        xy = np.asarray([x, y]).T
        self.velocities = feat.calculate_distance_vector(xy)

    def test_ivt_1_1(self):
        # fixation and saccade
        s, f = ivt(self.velocities, .01, min_fix_duration=1, sample_rate=1)
        self.assertTrue(np.allclose(list(s), [(3, 4)]))
        self.assertTrue(np.allclose(list(f), [(0, 2)]))

    def test_ivt_dur_equal(self):
        # min duration exactly equal to duration
        s, f = ivt(self.velocities, .01, min_fix_duration=3, sample_rate=1)
        self.assertTrue(np.allclose(list(s), [(3, 4)]))
        self.assertTrue(np.allclose(list(f), [(0, 2)]))

    def test_ivt_dur_larger(self):
        # min duration larger than smallest fixation -> only one saccade
        s, f = ivt(self.velocities, .01, min_fix_duration=4, sample_rate=1)
        self.assertTrue(np.allclose(list(s), [(0, 4)]))
        self.assertEqual(len(list(f)), 0)

        # short fixation at the end
        s, f = ivt(np.flip(self.velocities, 0), .01, min_fix_duration=4, sample_rate=1)
        self.assertTrue(np.allclose(list(s), [(0, 4)]))
        self.assertEqual(len(list(f)), 0)

    def test_ivt_single(self):
        # only saccade
        #    0  1  2  3  4  5
        x = [1, 2, 3, 4, 5, 6]
        y = [1, 2, 3, 4, 5, 6]
        xy = np.asarray([x, y]).T
        velocities = feat.calculate_distance_vector(xy)
        s, f = ivt(velocities, .01, min_fix_duration=1, sample_rate=1)
        self.assertTrue(np.allclose(list(s), [(0, 4)]))
        self.assertEqual(len(list(f)), 0)

        # only fixx
        #    0  1  2  3
        x = [0, 0, 0, 0]
        y = [0, 0, 0, 0]
        xy = np.asarray([x, y]).T
        velocities = feat.calculate_distance_vector(xy)
        s, f = ivt(velocities, .01, min_fix_duration=1, sample_rate=1)
        self.assertEqual(len(list(s)), 0)
        self.assertTrue(np.allclose(list(f), [(0, 2)]))

    def test_ivt_complex(self):
        # saccade in the middle
        #    0  1  2  3  4  5  6  7
        x = [0, 0, 0, 3, 4, 0, 0, 0]
        y = [0, 0, 0, 3, 4, 0, 0, 0]
        xy = np.asarray([x, y]).T
        velocities = feat.calculate_distance_vector(xy)
        s, f = ivt(velocities, .01, min_fix_duration=1, sample_rate=1)
        self.assertTrue(np.allclose(list(s), [(2, 4)]))
        self.assertTrue(np.allclose(list(f), [(0, 1), (5, 6)]))

        # # saccade in the middle, fixations too short
        s, f = ivt(velocities, .01, min_fix_duration=3, sample_rate=1)
        self.assertEqual(len(list(f)), 0)
        self.assertTrue(np.allclose(list(s), [(0, 6)]))

    def test_ivt_sacc_start_end(self):
        # test sequences with uneven amount of fixations and saccades
        x = [3, 4, 0, 0, 0, 3, 4]
        y = [3, 4, 0, 0, 0, 3, 4]
        xy = np.asarray([x, y]).T
        velocities = feat.calculate_distance_vector(xy)
        s, f = ivt(velocities, .01, min_fix_duration=1, sample_rate=1)
        self.assertTrue(np.allclose(list(s), [(0, 1), (4, 5)]))
        self.assertTrue(np.allclose(list(f), [(2, 3)]))
        
    def test_compare_ivt(self,verbose=False):

        n_samples = 10000

        parameters = [{'vel_threshold': np.sqrt(2)*100,
                       'min_fix_duration': 0,
                       'sample_rate': 1},
                      {'vel_threshold': 0,
                       'min_fix_duration': 0,
                       'sample_rate': 1},
                      {'vel_threshold': 0,
                       'min_fix_duration': n_samples,
                       'sample_rate': 1}
                      ]
        sample_start_result = [0, n_samples-1]
        sample_type_results = [['fixation'], ['saccade'], ['saccade']]

        trajectory = Trajectory(np.random.rand(n_samples, 2)*100, 'angle_deg')

        if verbose:
            print("IVT Test")

        def run_test(parameter, info):
            if verbose:
                print(info)
                start_time = time.time()
            sample_start1, sample_type1 = eye_movement_classification(trajectory,
                                                                      EyeMovementClassifier('IVT_old', parameter))
            if verbose:
                # noinspection PyUnboundLocalVariable
                print("--- old IVT took {} seconds".format(time.time() - start_time))
                start_time = time.time()
            sample_start2, sample_type2 = eye_movement_classification(trajectory,
                                                                      EyeMovementClassifier('IVT_new', parameter))
            if verbose:
                # noinspection PyUnboundLocalVariable
                print("--- new IVT took {} seconds".format(time.time() - start_time))

            try:
                self.assertEqual(sample_start1, sample_start2)
                self.assertEqual(sample_type1, sample_type2)
            except AssertionError:
                print('- IVTs failed')
                print(parameter)
                eye_movement_classification(trajectory, EyeMovementClassifier('IVT', parameter))
                raise
            if verbose:
                print('-- IVTs return the same')

            return sample_start1, sample_type1

        infos = ['all', 'noneThreshold', 'noneDuration']

        for ii in range(len(parameters)):
            sample_start, sample_type = run_test(parameters[ii],infos[ii])
            self.assertEqual(sample_start, sample_start_result)
            self.assertEqual(sample_type, sample_type_results[ii])

        parameters = [{'vel_threshold': 25,
                       'min_fix_duration': 0,
                       'sample_rate': 1},
                      {'vel_threshold': 0,
                       'min_fix_duration': 5,
                       'sample_rate': 1},
                      {'vel_threshold': 50,
                       'min_fix_duration': 5,
                       'sample_rate': 1},
                      {'vel_threshold': np.random.rand(1) * 200,
                       'min_fix_duration': np.random.rand(1) * 0.2,
                       'sample_rate': np.random.rand(1) * 1000}
                      ]

        infos = ['mixedThreshold', 'mixedDuration', 'mixed', 'random']

        for ii in range(len(parameters)):
            run_test(parameters[ii],infos[ii])
