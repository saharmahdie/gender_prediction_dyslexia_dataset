import timeit
import numpy as np

from schau_mir_in_die_augen.tests.test_features import TestFeatures
from schau_mir_in_die_augen.features import statistics, stat_names, basic_features_1d, \
    BasicFeatures1D
from schau_mir_in_die_augen.feature_extraction import all_features, all_features2
from schau_mir_in_die_augen.process.trajectory import Trajectory

unit_tests = TestFeatures()

repetitions = 1000
test_vector = np.random.rand(1000)
feature_names = [BasicFeatures1D.mean,
                 BasicFeatures1D.median,
                 BasicFeatures1D.max,
                 BasicFeatures1D.std,
                 BasicFeatures1D.skew,
                 BasicFeatures1D.kurtosis,
                 BasicFeatures1D.min,
                 BasicFeatures1D.var]

def old_features():
    dict(zip(stat_names('', True), statistics(test_vector)))

def new_features():
    basic_features_1d(test_vector, feature_names)

def test_compare_statistics():

    # make shure everything works correct
    unit_tests.test_compare_statistics()

    print('Compare Statistical Features')
    print(' old features take {} seconds'.format(timeit.timeit(
        'old_features()', number=repetitions, setup="from __main__ import old_features")))
    print(' new features take {} seconds'.format(timeit.timeit(
        'new_features()',  number=repetitions, setup="from __main__ import new_features")))

test_compare_statistics()


# noinspection PyRedeclaration
repetitions = 100
test_vector2x = np.random.rand(100,2)
screen_params = {'pix_per_mm': np.asarray([3.5443037974683542, 3.5443037974683542]),
                              'screen_dist_mm': 550.,
                              'screen_res': np.asarray([1680, 1050])}
trajectory = Trajectory(test_vector2x, sample_rate=1,
                        **Trajectory.screen_params_converter(screen_params))

def old_all_features():
    all_features(trajectory)

def new_all_features():
    all_features2(trajectory)

def test_compare_all_features():
    # make shure everything works correct
    # unit_tests.test_feature()

    print('Compare Feature calculating methods')
    print(' old method take {} seconds'.format(timeit.timeit(
        'old_all_features()', number=repetitions, setup="from __main__ import old_all_features")))
    print(' new method take {} seconds'.format(timeit.timeit(
        'new_all_features()', number=repetitions, setup="from __main__ import new_all_features")))

test_compare_all_features()
