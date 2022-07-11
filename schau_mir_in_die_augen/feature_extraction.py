""" Combination and application of the Features from features.py """

import numpy as np
import pandas as pd
from joblib import Memory
from scipy.stats import skew, kurtosis
from matplotlib import pyplot as plt
from enum import IntEnum

from timeit import default_timer as timer
import logging

import schau_mir_in_die_augen.process.conversion as converters
from schau_mir_in_die_augen.features import AllFeature, \
    basic_features_1d, BasicFeatures1D, \
    basic_features_2d, BasicFeatures2D, \
    extended_features_2d, ExtendedFeatures2D, \
    time_features, TimeFeatures, \
    history_features, HistoryFeatures, \
    statistics, \
    calculate_distance_vector, total_length, angle_between_first_and_last_points, \
    distance_between_first_and_last_points, calculate_dispersion, \
    distance_cog, stat_names, \
    general_gaze_points_features, acceleration_features, acceleration_features_names, \
    direction_changes_names, \
    direction_changes, micro_fixation, histogram, angle_among_3consecutive_Points, \
    angle_btw_2consecutive_points_in_vector, ngram_features, ngram_bins
from schau_mir_in_die_augen.process.trajectory import calculate_derivation, Trajectory
from schau_mir_in_die_augen.trajectory_classification.trajectory_split import EyeMovementClassifier, \
    EyeMovementClassification
from subprocess import call
from sklearn.tree import export_graphviz

import sys
sys.path.append('../')

# memory = Memory('~/tmp/smida-cache', verbose=0) if we are not in the admin pc account
memory = Memory('/tmp/smida-cache', verbose=0)
# memory.clear(warn=False)

class Data(IntEnum):
    data = 0                   # irrelevant
    complex = 1             # feature which need different data or other feature

    pixel_data = 10
    pixel_difference = 11
    pixel_velocity = 12         # difference * sample_rate
    pixel_velocity_change = 18  # difference(1D) * sample_rate => 1D => only work with dimension data
    pixel_difference2 = 13
    pixel_acceleration = 14     # difference2 * sample_rate

    deg_data = 20
    deg_difference = 21
    deg_velocity = 22
    deg_velocity_change = 28
    deg_difference2 = 23
    deg_acceleration = 24

    rad_data = 30
    rad_difference = 31
    rad_velocity = 32
    rad_velocity_change = 39
    rad_difference2 = 33
    rad_acceleration = 34

class Dimension(IntEnum):
    data = 00       # len, duration, ...?
    x_data = 10     # x component
    y_data = 20     # y component
    xy_data = 30    # x and y component
    xy_length = 40  # sqrt(x²+y²)

class ComplexFeatures(IntEnum):
    win_ratio = AllFeature.win_ratio
    avg_vel = AllFeature.avg_vel

def unpack_feature(feature_dict, prefix=None):

    if prefix is None:
        prefix = ''
        separator = ''
    else:
        separator = '_'
    unpacked_dict = {}

    for key in feature_dict:
        if isinstance(feature_dict[key], dict):
            unpacked_dict.update(unpack_feature(feature_dict[key], prefix+separator+key.name))
        else:
            unpacked_dict[prefix+separator+key.name] = feature_dict[key]
    return unpacked_dict


def trajectory_split_and_feature_cached(trajectory: Trajectory, method_name='IVT', method_parameter=None,
                                        classifier_parameter=None,
                                        feature_entries=None,
                                        omit_our: bool = False, omit_stats: bool = False,
                                        cached=True) -> pd.DataFrame:
    """ return features for Trajectory from cache if possible.
        If something have changed, thy will be calculated."""
    # todo: Trajectory slows this really down. Is it still?
    #   Maybe testing with some attributes removed (e.g. velocity - but it is used inside).
    #       Maybe a lighter Trajectory class could be used.
    #   Maybe structure could be changed so memory is used later.
    #   Otherwise could be necessary to put in only the important data and remove Trajectory.

    if cached:
        return memory.cache(trajectory_split_and_feature_method)(trajectory, EyeMovementClassifier(
            name=method_name,
            parameter=method_parameter, **classifier_parameter),
                                                                 feature_entries=feature_entries,
                                                                 omit_our=omit_our, omit_stats=omit_stats)
    else:
        # to test without cache
        return trajectory_split_and_feature_method(trajectory, EyeMovementClassifier(
            name=method_name,
            parameter=method_parameter, **classifier_parameter),
                                                   feature_entries=feature_entries,
                                                   omit_our=omit_our, omit_stats=omit_stats)


# noinspection PyDefaultArgument,PyUnreachableCode
def trajectory_split_and_feature_method(trajectory: Trajectory,
                                        eye_movement_classifier: EyeMovementClassifier = EyeMovementClassifier(
                                            name='IVT',
                                            parameter={'vel_threshold': 50, 'min_fix_duration': .1}),
                                        feature_entries=None,
                                        omit_our: bool = False, omit_stats: bool = False) -> pd.DataFrame:
    """ Generate feature vectors for all saccades and fixations in a trajectory

    :param trajectory: Trajectory class with gaze point data
    :param eye_movement_classifier: will be used to devide trajectory in different subsets
    :param feature_entries: to select a subset of features
    :param omit_our: don't calculate all feature
    :param omit_stats: don't calculate all feature
    :return: pandas.DataFrame with a row for each subset in trajectory and columnds depending on the featured
    """

    logger = logging.getLogger(__name__)
    logger.info('RUN trajectory_split_and_feature')
    logger.debug('Parameter: Classifier {cla_name}:{cla_parameter}, Feature {feat}, omit {omit}'.format(
        cla_name=eye_movement_classifier.name,
        cla_parameter=eye_movement_classifier.parameter,
        feat=feature_entries, omit=[omit_our, omit_stats]))

    # threshold and duration from the paper
    emc = EyeMovementClassification(trajectories=trajectory,
                                    subset_classifier=eye_movement_classifier)
    sample_starts, sample_types = emc.do_classification(0, 0)

    logger.debug('Trajectory Split returned {subs} subsets:\n {info}'.format(
        subs=len(sample_types), info=emc.get_type_overview(0, 0)))

    if "saccade" not in sample_types:
        print("no saccades")
    if "fixation" not in sample_types:
        print("no fixations")

    if feature_entries is None:
        trajectory.convert_to('pixel')
        logger.info('use old features')

    features = calculate_features(trajectory=trajectory,
                                  sample_starts=sample_starts, sample_types=sample_types,
                                  feature_entries=feature_entries,
                                  omit_our=omit_our, omit_stats=omit_stats)

    # remove rows for ignored subsets (shorter than 4)
    # todo: should this be here, or better outside?
    features = features[features['sample_type'] != 'ignored']

    logger.info('END trajectory_split_and_feature')

    return features


def calculate_features_cached(trajectory: Trajectory, sample_starts: list, sample_types: list, feature_entries=None,
                              omit_our: bool = False, omit_stats: bool = False, cached:bool = True) -> pd.DataFrame:
    """ return features for multiple subsets of Trajectory and different classifiers
        If something have changed, thy will be calculated.
    todo: Is trajectory slowing this down? """

    if cached:
        return memory.cache(calculate_features)(trajectory=trajectory, sample_starts=sample_starts,
                                                sample_types=sample_types, feature_entries=feature_entries,
                                                omit_our=omit_our, omit_stats=omit_stats)
    else:
        # to test without cache
        return calculate_features(trajectory=trajectory, sample_starts=sample_starts, sample_types=sample_types,
                                  feature_entries=feature_entries, omit_our=omit_our, omit_stats=omit_stats)


def calculate_features(trajectory: Trajectory, sample_starts: list, sample_types: list, feature_entries=None,
                       omit_our: bool = False, omit_stats: bool = False) -> pd.DataFrame:
    """ calculate features for multiple subsets of Trajectory and different classifiers

    :param trajectory: xy data in Trajectroy class
    :param sample_starts: starts/ends of subsets - we assume a continous set
    :param sample_types: type of subsets
    :param feature_entries: selection of features
    :param omit_our:
    :param omit_stats:
    :return: pandas.DataFrame with rows for every entry in sample_types and columns depending on the features.
             One column will be 'sample_type' and store 'ignored' where calculation was skipped (Values are NaN instead)
    """
    logger = logging.getLogger(__name__)
    assert len(sample_starts) == len(sample_types) + 1

    features = []
    for sample_type in list(set(sample_types)):
        # deal each typ together (prev_part is for same type)

        logger.info('Getting features for type: "{}"'.format(sample_type))

        # skip short samples
        if sample_type[:5] == 'short':
            # todo: maybe add NaN as feature to keep size?
            logger.debug('Skipped features for type: "{}"'.format(sample_type))
            continue

        sample_slices = [(slice(start, end), idd)
                         for start, end, (idd, typ) in zip(sample_starts[:-1], sample_starts[1:], enumerate(sample_types))
                         if typ == sample_type]

        prev_part = slice(0, 0)
        for part, sample_id in sample_slices:

            # some features need at least 4 points
            if part.stop - part.start < 4:
                # todo: i don't like this. Maybe it should be done elsewhere
                # i think the eye movement classification algorithm should prevent this.
                all_feats = {'sample_type': 'ignored', 'sample_id': sample_id}
            else:
                all_feats = calculate_features_subset(trajectory=trajectory,
                                                      use_slice=part, prev_slice=prev_part,
                                                      feature_entries=feature_entries,
                                                      omit_our=omit_our, omit_stats=omit_stats)
                all_feats['sample_type'] = sample_type
                all_feats['sample_id'] = sample_id
                prev_part = part

            features.append(all_feats)

    features = pd.DataFrame(features)

    features.set_index(keys=features['sample_id']+1, drop=True, inplace=True)
    features.drop(columns=['sample_id'], inplace=True)
    features.sort_index(inplace=True)

    logger.debug('{rows} rows with each {feat} features are returned.'.format(
        rows=features.shape[0], feat=features.shape[1]-1))  # one column for sample_type

    return features


def calculate_features_subset(trajectory, use_slice: slice = None, prev_slice: slice = None,
                              feature_entries=None,
                              omit_our: bool = False, omit_stats: bool = False) -> dict:
    """ This method is only to decide which method will be used
        When feature_entries are given the new method will be used, otherwise the old.
        :return dict with features
    """

    if feature_entries is None:

        feature_array = all_features(trajectory=trajectory,
                                     part=use_slice, prev_part=prev_slice,
                                     omit_our=omit_our, omit_stats=omit_stats,
                                     timing=False, pd_frame=False)
    else:
        feature_array = get_features(trajectory=trajectory, feature_entries=feature_entries,
                                     part=use_slice, prev_part=prev_slice)

    return feature_array


def features_2d(xy, feature_names):
    """ Use length of 2D vector """

    feature_names_1d = []
    while len(feature_names):
        # noinspection PyTypeChecker
        if feature_names[0] in list(BasicFeatures1D):
            feature_names_1d.append(feature_names.pop(0))
        else:
            raise Exception('Type {type} of feature "{name}" is unknown'.format(
                type=type(feature_names[0]), name=feature_names[0].name))

    if len(feature_names_1d):
        feature_dict = {Dimension.x_length: basic_features_1d(
            np.sqrt(np.square(xy[0])+np.square(xy[1])), feature_names_1d)}
    else:
        feature_dict = {}

    return feature_dict


def features_xy(xy, feature_names):
    """ Try to implement new method
    @:param xy: 2d vector
    @:param feature_names: dict of Features
        type of feature and name of feature
        types: BasicFeatures1D
    @:param prefix: string
        start of name
    """
    feature_names_1d = []
    feature_names_2d = []
    while len(feature_names):
        # noinspection PyTypeChecker
        if feature_names[0] in list(BasicFeatures1D):
            feature_names_1d.append(feature_names.pop(0))
        elif feature_names[0] in list(BasicFeatures2D):
            feature_names_2d.append(feature_names.pop(0))
        else:
            raise Exception('Type {type} of feature "{name}" is unknown'.format(
                type=type(feature_names[0]), name=feature_names[0].name))

    if len(feature_names_1d):
        feature_dict = {Dimension.x_data: basic_features_1d(xy[0], feature_names_1d),
                        Dimension.y_data: basic_features_1d(xy[1], feature_names_1d)}
    else:
        feature_dict = {}
    if len(feature_names_2d):
        feature_dict.update({Dimension.xy_data: basic_features_2d(xy, feature_names_2d)})

    return feature_dict

def tuples2dict(tuple_list):
    """ Creates dictionary from first value of tuples of list of tuples
    @:param tuple_list: [(x1,x2,x3,...),(y1,y2,y3,...),....(x1,y2,z3),....)
    @:return dict: {x1:[(x2,x3),...,(y2,z3),...], y1:[(y2,y3),...],... }
    """

    dict_output = {}

    for tuple_entry in tuple_list:
        if tuple_entry[0] in dict_output:
            dict_output[tuple_entry[0]].append(tuple_entry[1:])
        else:
            dict_output[tuple_entry[0]] = [tuple_entry[1:]]

    return dict_output

def get_basic_features(data, prev_data, sample_rate, feature_entries):

    check_len = len(feature_entries)
    feature_entries = [feature_entry[0] for feature_entry in feature_entries if len(feature_entry) == 1]
    if len(feature_entries) != check_len:
        raise Exception("This should not be possible.")
        # Feature should be in in their own list. They are put there by tuples2dict.from
        # If there are multiple Features in the List, than the structure seems to be changed.

    feature_dict = {}

    # noinspection PyTypeChecker
    feature_dict.update(basic_features_1d(data,
                                          [feature_entry for feature_entry in feature_entries
                                           if feature_entry in list(BasicFeatures1D)]))
    # noinspection PyTypeChecker
    feature_dict.update(basic_features_2d(data,
                                          [feature_entry for feature_entry in feature_entries
                                           if feature_entry in list(BasicFeatures2D)]))
    # noinspection PyTypeChecker
    feature_dict.update(extended_features_2d(data,
                                             [feature_entry for feature_entry in feature_entries
                                              if feature_entry in list(ExtendedFeatures2D)]))
    # noinspection PyTypeChecker
    feature_dict.update(time_features(data, sample_rate,
                                      [feature_entry for feature_entry in feature_entries
                                       if feature_entry in list(TimeFeatures)]))
    # noinspection PyTypeChecker
    feature_dict.update(history_features(data, prev_data,
                                         [feature_entry for feature_entry in feature_entries
                                          if feature_entry in list(HistoryFeatures)]))

    if len(feature_dict) != check_len:
        print('Feature demanded: {feature_entries}'.format)
        print('Feature returned: {feature_dict}'.format)
        raise Exception("The amount of demanded ({}) and returned ({}) features are not the same.".format(
            check_len,len(feature_dict)))
        # Some Features got lost during gathering.
        # Got be overwritten (unlikely).
        # Presumable one method doesn't return one ore more demanded feature (yet, there is no error for this)

    return feature_dict

def deal_data(data, prev_data, sample_rate, feature_entries, verbose=False):

    if not prev_data.shape[0]:
        prev_data = np.ndarray([4,2]) * np.nan
    elif verbose and prev_data.shape[0] < 3:
        print('Prev Data has only {nodes} nodes, This could be a problem.'.format(nodes=prev_data.shape[0]))

    check_len = len(feature_entries)

    # first group the data
    dimension_dict = tuples2dict(feature_entries)

    # check every grup and gather features
    feature_dict = {}
    if Dimension.data in dimension_dict:
        feature_dict[Dimension.data] = get_basic_features(data, prev_data, sample_rate,
                                                          dimension_dict[Dimension.data])

    if Dimension.x_data in dimension_dict:
        feature_dict[Dimension.x_data] = get_basic_features(data[:, 0], prev_data[:, 0], sample_rate,
                                                            dimension_dict[Dimension.x_data])

    if Dimension.y_data in dimension_dict:
        feature_dict[Dimension.y_data] = get_basic_features(data[:, 1], prev_data[:, 1], sample_rate,
                                                            dimension_dict[Dimension.y_data])

    if Dimension.xy_data in dimension_dict:
        feature_dict[Dimension.xy_data] = get_basic_features(data, prev_data, sample_rate,
                                                             dimension_dict[Dimension.xy_data])

    if Dimension.xy_length in dimension_dict:
        feature_dict[Dimension.xy_length] = get_basic_features(np.sqrt(np.square(data[:, 0]) + np.square(data[:, 1])),
                                                               np.sqrt(np.square(prev_data[:,0])+np.square(prev_data[:,1])),
                                                               sample_rate,
                                                               dimension_dict[Dimension.xy_length])

    if sum([len(feature_dict[subset]) for subset in feature_dict]) != check_len:
        raise Exception("This should not be possible.")
        # Some Dimensions got lost during gathering.
        # Got be overwritten (unlikely).
        # Presumable one method doesn't return one ore more demanded feature (yet, there is no error for this)

    return feature_dict

def deal_complex_data(feature_dict, feature_entries):

    check_len = len(feature_entries)
    feature_entries = [feature_entry[0] for feature_entry in feature_entries if len(feature_entry) == 1]
    if len(feature_entries) != check_len:
        raise Exception("This should not be possible.")
        # Feature should be in in their own list. They are put there by tuples2dict.from
        # If there are multiple Features in the List, than the structure seems to be changed.

    if ComplexFeatures.win_ratio in feature_entries:
        # max(angular_velocity) / duration
        feature_dict[ComplexFeatures.win_ratio] = \
            feature_dict[Data.deg_velocity][Dimension.xy_length][BasicFeatures1D.max] \
            / feature_dict[Data.data][Dimension.data][TimeFeatures.duration]
    if ComplexFeatures.avg_vel in feature_entries:
        # path_len=total_distance / duration
        feature_dict[ComplexFeatures.avg_vel] = \
            feature_dict[Data.pixel_difference][Dimension.xy_length][BasicFeatures1D.sum] \
            / feature_dict[Data.data][Dimension.data][TimeFeatures.duration]

    return feature_dict

def absolute(data_xy):
    return np.sqrt(np.square(data_xy[:, 0])+np.square(data_xy[:, 1]))

def gather_features(trajectory: Trajectory, feature_entries, part: slice = None, prev_part: slice = None):
    """ These method returns features for one subset """

    # todo: Implement Trajectory deeper => use Data types for selection.
    # todo: Implement Trajectory deeper => give outputs for all datatypes

    if not isinstance(part,slice):
        part = slice(part)
    if not isinstance(prev_part, slice):
        if prev_part is None:
            prev_part = slice(0)
        else:
            prev_part = slice(prev_part)

    slice_actual = (part, slice(None))
    slice_previous = (prev_part, slice(None))

    # first group the data
    data_dict = tuples2dict(feature_entries)

    # noinspection PyTypeChecker
    if any([data not in list(Data) for data in data_dict]):
        # noinspection PyTypeChecker
        raise Exception('Got unknown datatype: "{}"'.format([data for data in data_dict if data not in list(Data)]))

    # check every group and gather features
    feature_dict = {}

    def deal_data_helper(data, data_act, prev_data):
        nonlocal feature_dict
        if data in data_dict:
            feature_dict[data] = deal_data(data_act, prev_data, trajectory.sample_rate, data_dict.pop(data))

    # any data #
    ############
    deal_data_helper(Data.data, trajectory.xy[slice_actual], trajectory.xy[slice_previous])

    # shifted_pixel_data #
    ######################
    shifted_pixel_data = trajectory.get_trajectory('pixel_shifted')[slice_actual]
    prev_shifted_pixel_data = trajectory.get_trajectory('pixel_shifted')[slice_previous]

    deal_data_helper(Data.pixel_data, shifted_pixel_data, prev_shifted_pixel_data)

    if any([entry in data_dict for entry in [Data.pixel_difference,
                                             Data.pixel_velocity, Data.pixel_velocity_change,
                                             Data.pixel_difference2,
                                             Data.pixel_acceleration]]):

        pixel_difference = shifted_pixel_data[1:, :] - shifted_pixel_data[:-1, :]
        prev_pixel_difference = prev_shifted_pixel_data[1:, :] - prev_shifted_pixel_data[:-1, :]

        deal_data_helper(Data.pixel_difference, pixel_difference, prev_pixel_difference)

        if Data.pixel_velocity in data_dict or Data.pixel_velocity_change in data_dict:
            pixel_velocity = pixel_difference * trajectory.sample_rate
            prev_pixel_velocity = prev_pixel_difference * trajectory.sample_rate
            deal_data_helper(Data.pixel_velocity, pixel_velocity, prev_pixel_velocity)

            if Data.pixel_velocity_change in data_dict:

                deal_data_helper(Data.pixel_velocity_change,
                                 calculate_derivation(absolute(pixel_velocity), trajectory.sample_rate),
                                 calculate_derivation(absolute(prev_pixel_velocity), trajectory.sample_rate))

        if any([entry in data_dict for entry in [Data.pixel_difference2,
                                                 Data.pixel_acceleration]]):
            pixel_difference2 = pixel_difference[1:, :] - pixel_difference[:-1, :]
            prev_pixel_difference2 = prev_pixel_difference[1:, :] - prev_pixel_difference[:-1, :]

            deal_data_helper(Data.pixel_difference2, pixel_difference2, prev_pixel_difference2)

            if Data.pixel_acceleration in data_dict:
                pixel_acceleration = pixel_difference2 * trajectory.sample_rate * trajectory.sample_rate
                prev_pixel_acceleration = prev_pixel_difference2 * trajectory.sample_rate * trajectory.sample_rate
                deal_data_helper(Data.pixel_acceleration, pixel_acceleration, prev_pixel_acceleration)

    # rad data #
    ############
    if any([entry in data_dict for entry in [Data.rad_data,
                                             Data.rad_difference,
                                             Data.rad_velocity,
                                             Data.rad_difference2,
                                             Data.rad_acceleration]]):

        raise Exception('There is work to do')  # todo: this function in total should be possible to made by a template

    # deg data #
    ############
    if any([entry in data_dict for entry in [Data.deg_data,
                                             Data.deg_difference,
                                             Data.deg_velocity, Data.deg_velocity_change,
                                             Data.deg_difference2,
                                             Data.deg_acceleration]]):

        deg_data = trajectory.get_trajectory('angle_deg')[slice_actual]
        prev_deg_data = trajectory.get_trajectory('angle_deg')[slice_previous]

        deal_data_helper(Data.deg_data, deg_data, prev_deg_data)

        if any([entry in data_dict for entry in [Data.deg_difference,
                                                 Data.deg_velocity,
                                                 Data.deg_difference2,
                                                 Data.deg_acceleration]]):

            deg_difference = deg_data[1:, :] - deg_data[:-1, :]
            prev_deg_difference = prev_deg_data[1:, :] - prev_deg_data[:-1, :]

            deal_data_helper(Data.deg_difference, deg_difference, prev_deg_difference)

            if Data.deg_velocity in data_dict or Data.deg_velocity_change in data_dict:
                deg_velocity = deg_difference * trajectory.sample_rate
                prev_deg_velocity = prev_deg_difference * trajectory.sample_rate
                deal_data_helper(Data.deg_velocity, deg_velocity, prev_deg_velocity)

                if Data.deg_velocity_change in data_dict:
                    deal_data_helper(Data.deg_velocity_change,
                                     calculate_derivation(absolute(deg_velocity), trajectory.sample_rate),
                                     calculate_derivation(absolute(prev_deg_velocity), trajectory.sample_rate))

            if any([entry in data_dict for entry in [Data.deg_difference2,
                                                     Data.deg_acceleration]]):
                deg_difference2 = deg_difference[1:, :] - deg_difference[:-1, :]
                prev_deg_difference2 = prev_deg_difference[1:, :] - prev_deg_difference[:-1, :]

                deal_data_helper(Data.deg_difference2, deg_difference2, prev_deg_difference2)

                if Data.deg_acceleration in data_dict:
                    deg_acceleration = deg_difference2 * trajectory.sample_rate * trajectory.sample_rate
                    prev_deg_acceleration = prev_deg_difference2 * trajectory.sample_rate * trajectory.sample_rate
                    deal_data_helper(Data.deg_acceleration, deg_acceleration, prev_deg_acceleration)

    # Higher Feature #
    ##################

    if Data.complex in data_dict:
        feature_dict = deal_complex_data(feature_dict, data_dict.pop(Data.complex))

    if len(data_dict):
        raise Exception('data_dict should be empty by now! Content: "{}"'.format(data_dict))

    return feature_dict

def translate_feature_list(feature_names):

    new_feature_names = set()

    def basic_features_1d_helper(feature_function:str, feature_dimension:str, feature_type=None):

        # determine data type
        if feature_type is None:
            data = Data.pixel_data
        elif len(feature_type) == 7:
            if feature_type == 'ang_vel':
                data = Data.deg_velocity
            elif feature_type == 'ang_acc':
                data = Data.deg_velocity_change
            else:
                raise Exception("Should not happen")
        else:
            if feature_type == 'vel':
                data = Data.pixel_velocity
            elif feature_type == 'acc':
                data = Data.pixel_acceleration
            elif feature_type == 'ang':
                data = Data.deg_data
            else:
                raise Exception("Should not happen")

        # determine dimension
        if feature_dimension == '_x':
            dimension = Dimension.x_data
        elif feature_dimension == '_y':
            dimension = Dimension.y_data
        elif feature_dimension == '_l':
            dimension = Dimension.xy_length
        elif feature_dimension == '_d':
            dimension = Dimension.data
        else:
            raise Exception("Should not happen")

        # determine function
        if feature_function == 'mean':
            function = BasicFeatures1D.mean
        elif feature_function == 'median':
            function = BasicFeatures1D.median
        elif feature_function == 'std':
            function = BasicFeatures1D.std
        elif feature_function == 'var':
            function = BasicFeatures1D.var
        elif feature_function == 'min':
            function = BasicFeatures1D.min
        elif feature_function == 'max':
            function = BasicFeatures1D.max
        elif feature_function == 'skew':
            function = BasicFeatures1D.skew
        elif feature_function == 'kurtosis':
            function = BasicFeatures1D.kurtosis
        else:
            raise Exception("Should not happen")

        return data, dimension, function

    for name in feature_names:
        if name in ['std_x', 'std_y', 'skew_x', 'skew_y', 'kurtosis_x', 'kurtosis_y']:
            new_feature_names.add(basic_features_1d_helper(name[:-2], name[-2:]))
        elif name in ['vel_x_mean', 'vel_x_median', 'vel_x_max', 'vel_x_std',
                      'vel_x_skew', 'vel_x_kurtosis', 'vel_x_min', 'vel_x_var',
                      'vel_y_mean', 'vel_y_median', 'vel_y_max', 'vel_y_std',
                      'vel_y_skew', 'vel_y_kurtosis', 'vel_y_min', 'vel_y_var',
                      'acc_x_mean', 'acc_x_median', 'acc_x_max', 'acc_x_std',
                      'acc_x_skew', 'acc_x_kurtosis', 'acc_x_min', 'acc_x_var',
                      'acc_y_mean', 'acc_y_median', 'acc_y_max', 'acc_y_std',
                      'acc_y_skew', 'acc_y_kurtosis', 'acc_y_min', 'acc_y_var']:
            new_feature_names.add(basic_features_1d_helper(name[6:], name[3:5], name[:3]))
        elif name in ['ang_vel_mean', 'ang_vel_median', 'ang_vel_max', 'ang_vel_std',
                      'ang_vel_skew', 'ang_vel_kurtosis', 'ang_vel_min', 'ang_vel_var']:
            new_feature_names.add(basic_features_1d_helper(name[8:], '_l', name[:7]))
        elif name in ['ang_acc_mean', 'ang_acc_median', 'ang_acc_max', 'ang_acc_std',
                      'ang_acc_skew', 'ang_acc_kurtosis', 'ang_acc_min', 'ang_acc_var']:
            new_feature_names.add(basic_features_1d_helper(name[8:], '_d', name[:7]))

        elif name == 'duration':
            new_feature_names.add((Data.data,
                                   Dimension.data,
                                   TimeFeatures.duration))
        elif name == 'path_len':
            new_feature_names.add((Data.pixel_difference,
                                   Dimension.xy_length,
                                   BasicFeatures1D.sum))
        elif name == 'angle_prev_win':
            new_feature_names.add((Data.pixel_difference,
                                   Dimension.xy_data,
                                   HistoryFeatures.diff_angle_first_last))
        elif name == 'dist_prev_win':
            new_feature_names.add((Data.pixel_data,
                                   Dimension.xy_data,
                                   HistoryFeatures.distance_cog))
        elif name == 'dispersion':
            new_feature_names.add((Data.pixel_data,
                                   Dimension.xy_data,
                                   BasicFeatures2D.dispersion))
        elif name == 'avg_vel':
            # second level Feature: path_len=total_distance / duration
            new_feature_names.add((Data.pixel_difference, Dimension.xy_length, BasicFeatures1D.sum))
            new_feature_names.add((Data.data, Dimension.data, TimeFeatures.duration))
            new_feature_names.add((Data.complex, ComplexFeatures.avg_vel))
        elif name == 'win_ratio':
            # second level Feature: np.max(angular_vel) / duration
            new_feature_names.add((Data.deg_velocity, Dimension.xy_length, BasicFeatures1D.max))
            new_feature_names.add((Data.data, Dimension.data, TimeFeatures.duration))
            new_feature_names.add((Data.complex, ComplexFeatures.win_ratio))
        elif name == 'win_angle':
            new_feature_names.add((Data.pixel_data,
                                   Dimension.xy_data,
                                   ExtendedFeatures2D.angle_first_last))
        elif name == 'win_amplitude':
            new_feature_names.add((Data.pixel_data,
                                   Dimension.xy_data,
                                   ExtendedFeatures2D.distance_first_last))
        elif name == 'spatial_density':
            # ignore ... todo ...
            print('spatial_density not included')
        else:
            raise Exception('test')

    return new_feature_names

def get_features(trajectory: Trajectory, feature_entries, part: slice = None, prev_part: slice = None) -> dict:
    """ Return dict with features for one subset

    :param trajectory: Trajectory
        Gaze points and information about experiment (e.g. sample rate, screen size)
    :param feature_entries: [(Data,Dimension,Feature), (Data,Dimension,Feature), .... ]
        List with features you want to calculate
    :param part: slice
        Use this, if you want to use only a part of Trajectory.
    :param prev_part: slice
        Fore some features the previous part is necessarry.
    :return dict with featurenames and values
        You can change it into pandas.Dataframe by
        >>> pd.DataFrame(get_features(...), index=[0])
    """

    feature_dict = gather_features(trajectory=trajectory, feature_entries=feature_entries,
                                   part=part, prev_part=prev_part)

    unpacked_dict = unpack_feature(feature_dict)

    # fix NaN values
    for feature in unpacked_dict:
        if np.isnan(unpacked_dict[feature]):
            # todo: i don't like this!
            unpacked_dict[feature] = 0

    return unpacked_dict


def all_features2(trajectory: Trajectory, part: slice = None, prev_part: slice = None,
                  omit_stats=False, omit_our=True) -> dict:
    """ Calculate all features for a given trajectory subset.
    :param trajectory:
    :param part:
    :param prev_part:
    :param omit_stats:
    :param omit_our:
    :return dict with featurenames and values
        You can change it into pandas.Dataframe by
        >>> pd.DataFrame(all_features2(...), index=[0])
    """

    new_feature_list = translate_feature_list(all_subset(omit_stats=omit_stats, omit_our=omit_our))

    return get_features(trajectory=trajectory, feature_entries=new_feature_list, part=part, prev_part=prev_part)

def all_features(trajectory: Trajectory, part: slice = None, prev_part: slice = None,
                 omit_stats=False, omit_our=True, timing: bool = False, pd_frame: bool = True):
    """
    Calculate all features for a given trajectory subset.

    Input:
    :param trajectory: Trajectory
        2D array of gaze points (x,y) and more Information, like: sampleRate: float
        Sample rate of tracker in Hz
    :param part: slice
        actual part to deal with
    :param prev_part: slice
        2D array of last saccade's gaze points (x,y)
    :param omit_stats:
    :param omit_our:
    :param timing: print timing
    :param pd_frame: return features in dataframe

    Returns: dataframe
        1D list of numerical features with feature names in table header

    """
    if timing:
        start_time = timer()

    xy = trajectory.get_trajectory('pixel_shifted', data_slice=part)
    prev_xy = trajectory.get_trajectory('pixel_shifted', data_slice=prev_part)

    bio_ds_flag = False
    if trajectory.user != None:
        Bio_DS_age_df = pd.read_csv('../data/data_cleaned_biometrics_ds/ParticipantInfomation.csv')
        for name in Bio_DS_age_df['Name']:
            if name == trajectory.user:
                bio_ds_flag = True
        if bio_ds_flag:    # checks the number of users in trajectory against the number of users in BiometricDS
            bio_DS_age_feature = (Bio_DS_age_df.loc[Bio_DS_age_df['Name'] == trajectory.user, ['Age']].Age.item())

    bio_ds_flag = False
    if trajectory.user != None:
        Bio_DS_spatial_density_df = pd.read_csv('../data/data_cleaned_biometrics_ds/spatial_density_feature_biometricDS.csv')
        for name in Bio_DS_spatial_density_df['user']:
            if name == trajectory.user:
                bio_ds_flag = True
        if bio_ds_flag:
            features_name_bioDS = Bio_DS_spatial_density_df.columns
            bio_DS_sd_feature = Bio_DS_spatial_density_df.loc[Bio_DS_spatial_density_df['user'] == trajectory.user]
            sd_feature_dict = dict()
            for sd_col in features_name_bioDS:
                sd_feature_dict[sd_col] = bio_DS_sd_feature[sd_col]

    # Hotfix for Sahars Changes. ToDo: We should discuss this ...
    if False:
        # to add spatial_density feature
        spatial_density_df = pd.read_csv(
            '/home/sahar/PycharmProjects/smida2/scripts/results/last_experments/all185users_spatial_density.csv')
        spatial_density_feature = (spatial_density_df.loc[spatial_density_df['user'] == trajectory.user, ['spatial_density']].spatial_density.item())
    else:
        spatial_density_feature = 0
    if timing:
        trajectory_time = timer()

    sample_rate = trajectory.sample_rate

    duration = xy.shape[0] / sample_rate  # calculate each saccadic duration

    # features in screen space
    total_distance = total_length(calculate_distance_vector(xy))
    win_angle = angle_between_first_and_last_points(xy)

    # angular features
    xy_angles = converters.convert_shifted_pixel_coordinates_to_angles_deg(xy, **trajectory.screen_params)
    angular_vel = np.linalg.norm(calculate_derivation(xy_angles, sample_rate), axis=1)

    if len(prev_xy) > 0:
        angle_with_previous = win_angle - angle_between_first_and_last_points(prev_xy)
        distance_from_previous = distance_cog(xy, prev_xy)
    else:
        angle_with_previous = 0
        distance_from_previous = 0

    if bio_ds_flag == True:
        features = {'duration': duration,
                    'std_x': np.std(xy[:, 0]),
                    'std_y': np.std(xy[:, 1]),
                    'path_len': total_distance,
                    'angle_prev_win': angle_with_previous,
                    'dist_prev_win': distance_from_previous,
                    'skew_x': skew(xy[:, 0]),
                    'skew_y': skew(xy[:, 1]),
                    'kurtosis_x': kurtosis(xy[:, 0]),
                    'kurtosis_y': kurtosis(xy[:, 1]),
                    'dispersion': calculate_dispersion(xy),
                    'avg_vel': total_distance / duration,
                    'win_ratio': np.max(angular_vel) / duration,
                    'win_angle': win_angle,
                    'win_amplitude': distance_between_first_and_last_points(xy),
                    'spatial_density': spatial_density_feature,
                    'Age': bio_DS_age_feature
                    }
        features.update(sd_feature_dict)
    else:
        features = {'duration': duration,
                    'std_x': np.std(xy[:, 0]),
                    'std_y': np.std(xy[:, 1]),
                    'path_len': total_distance,
                    'angle_prev_win': angle_with_previous,
                    'dist_prev_win': distance_from_previous,
                    'skew_x': skew(xy[:, 0]),
                    'skew_y': skew(xy[:, 1]),
                    'kurtosis_x': kurtosis(xy[:, 0]),
                    'kurtosis_y': kurtosis(xy[:, 1]),
                    'dispersion': calculate_dispersion(xy),
                    'avg_vel': total_distance / duration,
                    'win_ratio': np.max(angular_vel) / duration,
                    'win_angle': win_angle,
                    'win_amplitude': distance_between_first_and_last_points(xy),
                    'spatial_density': spatial_density_feature
                    }

    velocity_xy = calculate_derivation(xy, sample_rate)
    vel_x, vel_y = velocity_xy[:, 0], velocity_xy[:, 1]

    angular_acc = calculate_derivation(angular_vel, sample_rate)
    acc_x = calculate_derivation(vel_x, sample_rate)
    acc_y = calculate_derivation(vel_y, sample_rate)

    if not omit_stats:
        features.update(**dict(zip(stat_names('vel_x', True), statistics(vel_x))),
                        **dict(zip(stat_names('vel_y', True), statistics(vel_y))),
                        **dict(zip(stat_names('acc_x', True), statistics(acc_x))),
                        **dict(zip(stat_names('acc_y', True), statistics(acc_y))),
                        **dict(zip(stat_names('ang_vel', True), statistics(angular_vel))),
                        **dict(zip(stat_names('ang_acc', True), statistics(angular_acc))))

    if not omit_our:
        # TODO we could split this for x and y -> make this a function
        angular_vel_total = np.sum(angular_vel)
        angular_acc_total = np.sum(angular_acc)
        angular_acc_pos = angular_acc[angular_acc > 0]
        acc_pos_max = np.max(angular_acc_pos) if len(angular_acc_pos) else 0
        acc_pos_min = np.min(angular_acc_pos) if len(angular_acc_pos) else 0
        angular_acc_pos_diff = acc_pos_max - acc_pos_min
        angular_acc_pos_factor = acc_pos_max / acc_pos_min if acc_pos_min != 0 else 0

        angular_acc_neg = angular_acc[angular_acc < 0]
        acc_neg_max = np.max(angular_acc_neg) if len(angular_acc_neg) else 0
        acc_neg_min = np.min(angular_acc_neg) if len(angular_acc_neg) else 0
        angular_acc_neg_diff = acc_neg_max - acc_neg_min
        angular_acc_neg_factor = acc_neg_max / acc_neg_min if acc_neg_min != 0 else 0

        micro_fix_ = micro_fixation(xy, name_prefix='thresh_none')
        micro_fix_5 = micro_fixation(xy, 5, name_prefix='thresh_5')
        micro_fix_10 = micro_fixation(xy, 10, name_prefix='thresh_10')
        histogram_steps = 20
        angle_3points_list = angle_among_3consecutive_Points(xy)
        if len(angle_3points_list):  # prevent empty calls
            histogram_features = histogram(np.vstack(angle_3points_list), histogram_steps)
        else:
            histogram_features = [[None],[None]]

        # n-gram features
        ngram_directions_number = 8
        angle_2points_list = angle_btw_2consecutive_points_in_vector(xy)
        uni_ngram = ngram_features(ngram_bins(angle_2points_list, 8), 1, name_prefix='unigram')
        bi_ngram = ngram_features(ngram_bins(angle_2points_list, ngram_directions_number), 2, name_prefix='bigram')

        features.update({**dict(zip(acceleration_features_names("deacc"),
                                    acceleration_features(angular_acc[angular_acc < 0]))),
                         **dict(zip(acceleration_features_names("acc"),
                                    acceleration_features(angular_acc[angular_acc > 0]))),
                         **dict(zip(direction_changes_names(), direction_changes(xy))),
                         'angular_vel_total': angular_vel_total,
                         'angular_acc_neg_diff': angular_acc_neg_diff,
                         'angular_acc_neg_factor': angular_acc_neg_factor,
                         'angular_acc_pos_diff': angular_acc_pos_diff,
                         'angular_acc_pos_factor': angular_acc_pos_factor,
                         'angular_acc_total': angular_acc_total,
                         **dict(zip(uni_ngram[1], uni_ngram[0])),
                         **dict(zip(bi_ngram[1], bi_ngram[0])),
                         **dict(zip(histogram_features[1], histogram_features[0])),
                         **dict(zip(micro_fix_[1], micro_fix_[0])),
                         **dict(zip(micro_fix_5[1], micro_fix_5[0])),
                         **dict(zip(micro_fix_10[1], micro_fix_10[0])),
                         })

    if timing:
        calc_time = timer()

    if pd_frame:
        features = pd.DataFrame(features, index=[0])

    if timing:
        end_time = timer()
        # noinspection PyUnboundLocalVariable
        print('Trajectory: {tra:4f}s, Calculation: {cal:4f}, Dataframe {res:4f}s, Complete {com:4f} seconds.'.format(
            tra=trajectory_time-start_time,
            cal=calc_time-trajectory_time,
            res=end_time-calc_time,
            com=end_time-start_time))

    return features


# Selection of Features #
#########################

def all_subset(omit_stats=False,omit_our=True):
    return list(all_features(Trajectory(
        xy=np.random.rand(4,2), kind='pixel', sample_rate=1,
        pix_per_mm=np.asarray([10, 20]), screen_dist_mm=50, fov_x=300, fov_y=400),
        omit_stats=omit_stats, omit_our=omit_our))

def sac_subset(text=True):
    if text:
        fix_cols = ['dispersion'] \
                   + stat_names('ang_vel', False)[1:] \
                   + stat_names('ang_acc', False)[:-1] \
                   + ['std_x',
                      'std_y',
                      'path_len',
                      'angle_prev_win', 'dist_prev_win', 'win_ratio', 'win_angle', 'win_amplitude'] \
                   + stat_names('vel_x', False) \
                   + stat_names('vel_y', False) \
                   + stat_names('acc_x', False) \
                   + stat_names('acc_y', False)
    else:
        # RAN data
        vel_names = stat_names('vel_y', False)
        acc_names = stat_names('acc_y', False)
        fix_cols = ['dispersion'] \
                   + stat_names('ang_vel', False)[3:] \
                   + stat_names('ang_acc', False) \
                   + ['std_x',
                      'std_y',
                      'path_len',
                      'angle_prev_win', 'dist_prev_win', 'win_ratio', 'win_angle', 'win_amplitude'] \
                   + stat_names('vel_x', False) \
                   + vel_names[:4] + [vel_names[5]] \
                   + stat_names('acc_x', False) \
                   + acc_names[:2] + acc_names[3:]

    return fix_cols


def fix_subset(text=True):
    if text:
        fix_cols = ['std_y',
                    'path_len',
                    'angle_prev_win',
                    'dist_prev_win',
                    'skew_x',
                    'skew_y',
                    'kurtosis_y',
                    'dispersion',
                    'avg_vel']
    else:
        # RAN data
        fix_cols = ['duration',
                    'path_len',
                    'angle_prev_win',
                    'dist_prev_win',
                    'skew_x',
                    'skew_y',
                    'kurtosis_y',
                    'dispersion',
                    'avg_vel']

    return fix_cols


def paper_all_subset():
    fix_cols = ['duration',
                'path_len',
                'skew_x',
                'skew_y',
                'kurtosis_x',
                'kurtosis_y',
                'avg_vel',
                'std_x',
                'std_y',
                'angle_prev_win',
                'dist_prev_win',
                'win_ratio',
                'win_angle',
                'win_amplitude',
                'dispersion'] \
               + stat_names('ang_vel', False) \
               + stat_names('ang_acc', False) \
               + stat_names('vel_x', False) \
               + stat_names('vel_y', False) \
               + stat_names('acc_x', False) \
               + stat_names('acc_y', False)

    return fix_cols


# Selection Helper? #
#####################

def trajectory_features(xy, sample_rate, screen_params):
    general_gaze = general_gaze_points_features(xy, sample_rate, screen_params)

    features = {**dict(zip(general_gaze[1], general_gaze[0]))}
    return pd.DataFrame(features, index=[0])


def important_features(clf, x, n=5):  # n the number of the top features.
    # print("Important features from saccade", self.clf.feature_importances_)

    importances = clf.feature_importances_

    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    plot_variance = False
    if plot_variance:
        plt.figure()
        plt.title("Feature importances", fontsize=20)
        plt.bar(range(x.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(x.shape[1]), indices)
        plt.xlim([-1, x.shape[1]])
        # plt.show()

    feat_importances = pd.Series(importances, index=x.columns)
    feat_importances.nlargest(n).plot.bar()  # (kind='barh')
    # plt.show()
    print(feat_importances.nlargest(n))
    # plt.tick_params(axis='both', which='major', labelsize=18);
    plt.tick_params(axis='both', which='minor', labelsize=10)

    return feat_importances.nlargest(n).index.array


# Plot Decision Tree #
######################
# todo: Why is this here?

def plot_dt(estimator, graph_path, top_feat, target_data, filename):
    export_graphviz(estimator, out_file=graph_path+filename+'.dot',
                    feature_names=top_feat,
                    class_names=target_data,
                    rounded=True, proportion=False,
                    precision=2, filled=True)
    call(['dot', '-Tpng', graph_path + filename + '.dot', '-o', graph_path + filename + '.png', '-Gdpi=1050'])
