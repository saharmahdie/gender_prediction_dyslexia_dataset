""" Implementation of single features """

import math
import numpy as np
import scipy
from scipy.stats import kurtosis
from itertools import product
from numpy import inf
from enum import IntEnum
import collections

import schau_mir_in_die_augen.process.conversion as converters
from schau_mir_in_die_augen.process.trajectory import calculate_derivation


class AllFeature(IntEnum):

    # 1: basic
    nan = 10                # number of nan
    len = 11                # length of vector
    sum = 12                # sum of vector
    total = 12

    # 2: statistic
    min = 21                # minimal value
    max = 22                # maximal value
    dispersion = 23         # max-min
    mean = 24               # average
    median = 25             # median
    std = 26                # standard deviation
    var = 27                # variance
    skew = 28               # skewness
    kurtosis = 29           # kurtosis

    # 3: timing
    duration = 31           # sample / samplerate
    time = 31

    # 4: extended
    angle_first_last = 41   # angle between first and last
    distance_first_last = 42    # distance between first and last element

    # 5: historic
    distance_cog = 51
    diff_angle_first_last = 52  # difference between (angle between first and last)

    # 10: Complex Feature
    win_ratio = 101
    avg_vel = 102


class BasicFeatures1D(IntEnum):
    # feature should ignore nan if possible

    # 1: basic
    len = AllFeature.len
    sum = AllFeature.sum
    total = AllFeature.total
    nan = AllFeature.nan

    # 2: statistic
    min = AllFeature.min
    max = AllFeature.max
    dispersion = AllFeature.dispersion
    mean = AllFeature.mean
    median = AllFeature.median
    std = AllFeature.std
    var = AllFeature.var
    skew = AllFeature.skew
    kurtosis = AllFeature.kurtosis

    # 4:angular
    angle_first_last = AllFeature.angle_first_last

# noinspection PyDefaultArgument,PyTypeChecker
def basic_features_1d(vector, feature_names: list = list(BasicFeatures1D)):
    """ Try new statistics feature version """

    # prepare feature
    v_len, (v_min, v_max), v_mean, v_var, v_skew, v_kurt = scipy.stats.describe(vector, ddof=0)

    # put all asked feature in list
    if BasicFeatures1D.len in feature_names:
        feature_dict = {BasicFeatures1D.len: v_len}
    else:
        feature_dict = {}
    if BasicFeatures1D.sum in feature_names:
        feature_dict.update({BasicFeatures1D.sum: np.nansum(vector)})
    if BasicFeatures1D.nan in feature_names:
        feature_dict.update({BasicFeatures1D.sum: (np.isnan(vector))})
    if BasicFeatures1D.min in feature_names:
        feature_dict.update({BasicFeatures1D.min: v_min})
    if BasicFeatures1D.max in feature_names:
        feature_dict.update({BasicFeatures1D.max: v_max})
    if BasicFeatures1D.dispersion in feature_names:
        feature_dict.update({BasicFeatures1D.dispersion: v_max - v_min})
    if BasicFeatures1D.mean in feature_names:
        feature_dict.update({BasicFeatures1D.mean: v_mean})
    if BasicFeatures1D.median in feature_names:
        feature_dict.update({BasicFeatures1D.median: np.nanmedian(vector)})
    if BasicFeatures1D.std in feature_names:
        feature_dict.update({BasicFeatures1D.std: np.sqrt(v_var)})
    if BasicFeatures1D.var in feature_names:
        feature_dict.update({BasicFeatures1D.var: v_var})
    if BasicFeatures1D.skew in feature_names:
        feature_dict.update({BasicFeatures1D.skew: v_skew})
    if BasicFeatures1D.kurtosis in feature_names:
        feature_dict.update({BasicFeatures1D.kurtosis: v_kurt})

    return feature_dict

def stat_names(prefix="", extended=False):
    return ["{}_mean".format(prefix),
            "{}_median".format(prefix),
            "{}_max".format(prefix),
            "{}_std".format(prefix),
            "{}_skew".format(prefix),
            "{}_kurtosis".format(prefix)] \
           + (["{}_min".format(prefix),
               "{}_var".format(prefix)] if extended else [])


def statistics(v):
    """ statistical features function

    Input:
    v: ndarray
        1D float array
    extended: boolean
        to include min and var

    Return: list with 6 or 8 entries
         statistical features (M3S2K: Mean, Median, Max,STD ,Skewness, Kurtosis)
         statistical features extended (M3S2KMV: Mean, Median, Max,STD ,Skewness, Kurtosis, min, var)

    """

    if len(v) < 1:
        return [0] * len(stat_names(extended=True))

    assert len(v.shape) == 1 and v.shape[0] > 0

    _, (v_min, v_max), v_mean, v_var, v_skew, v_kurt = scipy.stats.describe(v, ddof=0)

    return [v_mean, np.median(v), v_max, np.sqrt(v_var), v_skew, v_kurt, v_min, v_var]


class BasicFeatures2D(IntEnum):
    len = AllFeature.len
    dispersion = AllFeature.dispersion

# noinspection PyDefaultArgument,PyTypeChecker
def basic_features_2d(xy, feature_names: list = list(BasicFeatures2D)):

    if BasicFeatures2D.len in feature_names:
        feature_dict = {BasicFeatures2D.len: xy.shape[0]}
    else:
        feature_dict = {}
    if BasicFeatures2D.dispersion in feature_names:
        feature_dict[BasicFeatures2D.dispersion] = calculate_dispersion(xy)

    return feature_dict

class TimeFeatures(IntEnum):
    # with sample_rate
    duration = AllFeature.duration

# noinspection PyDefaultArgument,PyTypeChecker
def time_features(xy, sample_rate, feature_names: list = list(TimeFeatures)):

    if TimeFeatures.duration in feature_names:
        feature_dict = {TimeFeatures.duration: xy.shape[0] / sample_rate}
    else:
        feature_dict = {}

    return feature_dict

class ExtendedFeatures2D(IntEnum):
    angle_first_last = AllFeature.angle_first_last
    distance_first_last = AllFeature.distance_first_last

# noinspection PyDefaultArgument,PyTypeChecker
def extended_features_2d(xy, feature_names: list = list(ExtendedFeatures2D)):

    if ExtendedFeatures2D.angle_first_last in feature_names:
        feature_dict = {ExtendedFeatures2D.angle_first_last:
                            angle_between_first_and_last_points(xy)}
    else:
        feature_dict = {}
    if ExtendedFeatures2D.distance_first_last in feature_names:
        feature_dict[ExtendedFeatures2D.distance_first_last] = distance_between_first_and_last_points(xy)

    return feature_dict

class HistoryFeatures(IntEnum):
    diff_angle_first_last = AllFeature.diff_angle_first_last
    distance_cog = AllFeature.distance_cog

# noinspection PyDefaultArgument,PyTypeChecker
def history_features(xy, prev_xy, feature_names: list = list(HistoryFeatures)):

    if HistoryFeatures.diff_angle_first_last in feature_names:
        feature_dict = {HistoryFeatures.diff_angle_first_last:
                            angle_between_first_and_last_points(xy) - angle_between_first_and_last_points(prev_xy)}
    else:
        feature_dict = {}
    if HistoryFeatures.distance_cog in feature_names:
        feature_dict[HistoryFeatures.distance_cog] = distance_cog(xy, prev_xy)

    return feature_dict

def calculate_dispersion(xy):
    """ dispersion function for spatial spread during a fixaton and saccade

    Input:
    xy: ndarray
        2D array of gaze points (x,y)

    Return: float
        dispersion

    """
    assert len(xy.shape) == 2 and xy.shape[1] == 2
    assert xy.shape[0] >= 1  # check if the input has at least 1 row

    return (np.max(xy[:, 0]) - np.min(xy[:, 0])) + (np.max(xy[:, 1]) - np.min(xy[:, 1]))


def angle_between_first_and_last_points(xy):
    """ Angle between two points in degrees

    Input:
    xy: ndarray
        2D array of gaze points (x,y)

    Return: float
        angle between the first and last point

    """
    assert len(xy.shape) == 2 and xy.shape[1] == 2
    assert xy.shape[0] >= 2  # check if the input has at least 2 rows

    return math.degrees(math.atan2(xy[-1, 0]-xy[0, 0], xy[-1, 1]-xy[0, 1]))


def angle_with_previous_fix(xyz, prev_xyz):
    """ Angle between current and previous points means in degrees

    Input:
    xy: ndarray
        3D array of gaze points (x,y,z)
    xy_prev: ndarray
        3D array of last saccade's gaze points (x,y,z)

    Return: float
        Angle between the AVG of the current points and the AVG of the previous points

    """
    assert xyz.shape[1] == 3
    assert prev_xyz.shape[1] == 3
    assert xyz.shape[0] >= 1       # check if the input has at least 1 row
    assert prev_xyz.shape[0] >= 1  # check if the input has at least 1 row

    point_a = np.mean(xyz, axis=0)
    point_b = np.mean(prev_xyz, axis=0)

    point_a = point_a / np.linalg.norm(point_a)
    point_b = point_b / np.linalg.norm(point_b)

    # due to rounding errors this can get larger than 1, which is not cool for arccos
    dp = min(1, np.dot(point_a, point_b))
    return math.degrees(np.arccos(dp))
    # return math.degrees(np.arccos(np.dot(point_a, point_b)/(np.linalg.norm(point_a)* np.linalg.norm(point_b))))


def angle_btw_2consecutive_points_in_vector(xy):
    """ Angles between each two consecutive points in vector of points

    Input:
    xy: ndarray
        2D array of gaze points (x,y)

    Return: ndarray
        1D array of angles in degrees with len = len(xy) - 1

     """
    assert len(xy.shape) == 2 and xy.shape[1] == 2
    assert xy.shape[0] >= 2  # check if the input has at least 2 rows

    diff = np.diff(xy, axis=0)
    angles = np.degrees(np.arctan2(diff[:, 0], diff[:, 1]))
    # zeros = np.where(np.sum(np.diff(xy, axis=0), axis=1) == 0)[0]   # nomovement case
    # angles[zeros] = 1000

    return angles


def angle_among_3consecutive_Points(xy):
    """ Angles for each three consecutive points in vector of points

     Input:
     xy: ndarray
        2D array of gaze points (x,y)

    Return: ndarray
        1D array of angles in degrees with len = len(xy) - 2

     """
    assert len(xy.shape) == 2 and xy.shape[1] == 2

    a = xy[:-2]
    b = xy[1:-1]
    c = xy[2:]
    ab = a - b
    bc = b - c
    # TODO here we use atan(y,x) and in other places atan(x,y)
    angle = np.degrees(np.arctan2(bc[:, 1], bc[:, 0]) - np.arctan2(ab[:, 1], ab[:, 0]))
    # assure range is [0, 360] and not negative
    # ToDo is this correct? there should be angles arround 0 and you bring them up to 360 !!!
    angle[angle < 0] += 360

    return angle


def calculate_distance_vector(xy):
    """ Distances to the predecessor
    Input:
    xy: ndarray
        2D array of gaze points (x,y)

    Returns: ndarray
        1D array of consecutive distances with len = len(xy) - 1

    """
    assert len(xy.shape) == 2 and xy.shape[1] == 2

    diff = xy[1:] - xy[:-1]
    sq = diff * diff

    return np.sqrt(sq[:, 0] + sq[:, 1])


def total_length(dis_vec):
    """
    Input:
    dis_vec: ndarray
        1D array of distances

    Returns: float
        Total distance

    """
    assert len(dis_vec.shape) == 1 and dis_vec.shape[0] > 0

    return np.sum(dis_vec)


def distance_between_first_and_last_points(xy):
    """ distance between two points

    Input:
    xy: ndarray
        2D array of gaze points (x,y)

    Reurns: float
        Distance between the first and last point

    """
    assert len(xy.shape) == 2 and xy.shape[1] == 2

    return np.sqrt(np.sum(np.square(xy[-1, :]-xy[0, :])))


def distance_cog(xy, prev_xy):
    """ Distance between the current centroid point in fixation or saccade with centroid point of the previous one
    Input:
    xy: ndarray
        2D array of gaze points (x,y)
    xy_prev: ndarray
        2D array of last saccade's gaze points (x,y)

    Returns: float
       Absolute distance
    """

    # ToDo: The name is somehow misleading. It depends on the input.

    assert len(xy.shape) == 2 and xy.shape[1] == 2
    assert len(prev_xy.shape) == 2 and prev_xy.shape[1] == 2

    point_a = np.mean(xy, axis=0)
    point_b = np.mean(prev_xy, axis=0)
    diff = point_b - point_a
    sq = diff * diff

    return np.sqrt(sq[0] + sq[1])


def general_gaze_points_features(xy, sample_rate, screen_params):
    distance_vector = calculate_distance_vector(xy)

    distance_stats = statistics(distance_vector)
    distance_total = total_length(distance_vector)

    speed_vector = np.multiply(distance_vector, sample_rate)

    speed_stats = statistics(speed_vector)
    speed_total = total_length(speed_vector)

    # acceleration from speed
    acceleration_vector = calculate_derivation(speed_vector, sample_rate)

    acc_vec_indx = np.where(acceleration_vector > 0)
    deacc_vec_indx = np.where(acceleration_vector < 0)
    acceleration_feats = acceleration_features(acceleration_vector[acc_vec_indx])
    deacceleration_feats = acceleration_features(acceleration_vector[deacc_vec_indx])

    angl_vec = converters.convert_shifted_pixel_coordinates_to_angles_deg(xy, **screen_params)
    angular_vel = np.linalg.norm(calculate_derivation(angl_vec, sample_rate), axis=1)

    angular_vel_stats = statistics(angular_vel)
    angular_vel_total = total_length(angular_vel)

    # angular acceleration from angular velocities
    angular_acceleration_vector = calculate_derivation(angular_vel, sample_rate)
    # TODO we took > and < 0 angular acc but we did not take the 0
    angular_acc_vec_indx = np.where(angular_acceleration_vector > 0)
    angular_deacc_vec_indx = np.where(angular_acceleration_vector < 0)
    angular_acceleration_feats = acceleration_features(angular_acceleration_vector[angular_acc_vec_indx])
    angular_deacceleration_feats = acceleration_features(angular_acceleration_vector[angular_deacc_vec_indx])

    dispersion = calculate_dispersion(xy)

    num_change_direction = direction_changes(xy)

    general_gaze_points_features_header = stat_names("traj_distance", True) \
                                          + ["traj_distance_total"] \
                                          + stat_names("traj_speed", True) \
                                          + ["traj_speed_total"] \
                                          + stat_names("traj_angular_vel", True) \
                                          + ["traj_angular_vel_total"] \
                                          + acceleration_features_names('traj_ang_acceleration') \
                                          + acceleration_features_names('traj_ang_de-acceleration') \
                                          + acceleration_features_names('traj_acceleration') \
                                          + acceleration_features_names('traj_de-acceleration') \
                                          + ["traj_dispersion"] \
                                          + direction_changes_names(prefix="traj")
    return distance_stats \
           + [distance_total] \
           + speed_stats \
           + [speed_total] \
           + angular_vel_stats \
           + [angular_vel_total] \
           + angular_acceleration_feats \
           + angular_deacceleration_feats \
           + acceleration_feats \
           + deacceleration_feats \
           + [dispersion] \
           + num_change_direction \
        , general_gaze_points_features_header


def direction_changes(xy):
    angle = angle_among_3consecutive_Points(xy)
    # ToDo maybe wrong? Returns always 2 elements less than len(xy)
    num_change_direction = len(angle)
    num_change_direction_threshold = []
    for i in range(1, 21):
        less_than = 360 - i
        greater_than = i
        num_change_direction_threshold.append(len(angle[np.where((angle > greater_than) & (angle < less_than))]))
    return [num_change_direction] + num_change_direction_threshold


def direction_changes_names(prefix=""):
    return ["{}_num_change_direction".format(prefix)] + ["{}_num_change_direction_threshold_{}".format(prefix, i) for i in range(1, 21)]


def acceleration_features(acc_vec):
    if len(acc_vec) == 0:
        acc_vec = np.array([0])
    # ToDo: This functions has nothing to do with acceleration. It's further extended statistics.

    acceleration_stats = statistics(acc_vec)
    acceleration_min = np.min(acc_vec)
    acceleration_max = np.max(acc_vec)
    acceleration_total = total_length(acc_vec)
    # ToDo: This returns only the number of list entries
    count_posative_acceleration = len(acc_vec)
    diff_max_min_acceleration = acceleration_max - acceleration_min
    # ToDo: I don't realy like that. Should return "inf" if possible, but 0 is missleading.
    #  Can we return -0 or something like that?
    if acceleration_min != 0:
        factor_max_min_acceleration = acceleration_max / acceleration_min
    else:
        factor_max_min_acceleration = 0

    return acceleration_stats + \
           [acceleration_total,
            count_posative_acceleration,
            diff_max_min_acceleration,
            factor_max_min_acceleration]

def acceleration_features_names(prefix=""):
    return stat_names('{}_feats'.format(prefix), True) + \
           ['{}_total'.format(prefix),
            '{}_count_positive'.format(prefix),
            '{}_diff_max_min'.format(prefix),
            '{}_factor_max_min'.format(prefix)]


def non_distributional_features(fix_vec, sampleRate, fix_dis_vec, rec_duration, fix_angular_vel_vec, fix_sac):
    num_fixation = len(fix_vec)

    fixation_rate = num_fixation / rec_duration
    fix_time_list = np.array([(item[1] - item[0]) / sampleRate for item in fix_vec])
    fix_dis_vec = np.array([np.sum(x) for x in fix_dis_vec])

    fix_time_list = np.where(fix_time_list == -inf, 0, fix_time_list)
    fix_dis_vec = np.where(fix_dis_vec == -inf, 0, fix_dis_vec)

    fix_speed_list = [item / fix_time_list[key] for key, item in enumerate(fix_dis_vec)]
    fix_speed_list = np.where(fix_speed_list == -inf, 0, fix_speed_list)

    fixation_time_stats = statistics(fix_time_list)
    total_fixation_time = sum(fix_time_list)

    fixation_distance_stats = statistics(fix_dis_vec)
    total_fixation_distance = sum(fix_dis_vec)

    fixation_speed_stats = statistics(fix_speed_list)
    total_fixation_speed = sum(fix_speed_list)

    fixation_angular_vel_stats = statistics(fix_angular_vel_vec)
    total_fixation_angular_vel = sum(fix_angular_vel_vec)

    # acceleration from speed
    fix_acc_list = calculate_derivation(fix_speed_list, sampleRate)

    acc_vec_indx = np.where(fix_acc_list > 0)
    deacc_vec_indx = np.where(fix_acc_list < 0)
    acceleration_feats = acceleration_features(fix_acc_list[acc_vec_indx])
    deacceleration_feats = acceleration_features(fix_acc_list[deacc_vec_indx])

    # angular acceleration
    fix_ang_acc_list = calculate_derivation(fix_angular_vel_vec, sampleRate)
    fix_ang_acc_list = np.where(fix_ang_acc_list == -inf, 0, fix_ang_acc_list)

    ang_acc_vec_indx = np.where(fix_ang_acc_list > 0)
    ang_deacc_vec_indx = np.where(fix_ang_acc_list < 0)
    angular_acceleration_feats = acceleration_features(fix_ang_acc_list[ang_acc_vec_indx])
    angular_deacceleration_feats = acceleration_features(fix_ang_acc_list[ang_deacc_vec_indx])

    non_distributional_fixation_features_headers = ["num_{}".format(fix_sac),
                                                    '{}_rate'.format(fix_sac)] \
                                                   + stat_names("{}_time_".format(fix_sac), True) \
                                                   + ["total_{}_time".format(fix_sac)] \
                                                   + stat_names("{}_distance_".format(fix_sac), True) \
                                                   + ["total_{}_distance".format(fix_sac)] \
                                                   + stat_names("{}_speed_".format(fix_sac), True) \
                                                   + ["total_{}_speed".format(fix_sac)] \
                                                   + stat_names("{}_angular_vel_".format(fix_sac), True) \
                                                   + ["total_{}_angular_vel".format(fix_sac)] \
                                                   + stat_names("{}_ang_acceleration_".format(fix_sac), True) \
                                                   + ["fix_ang_acceleration_total",
                                                      "fix_count_posative_ang_acceleration",
                                                      "fix_diff_max_min_ang_acceleration",
                                                      "fix_factor_max_min_ang_acceleration"] \
                                                   + stat_names("fixation_ang_deacceleration_", True) \
                                                   + ["fix_ang_deacceleration_total",
                                                      "fix_count_negative_ang_acceleration",
                                                      "fix_diff_max_min_ang_deacceleration",
                                                      "fix_factor_max_min_ang_deacceleration"] \
                                                   + stat_names("fixation_acceleration_", True) \
                                                   + ["fix_acceleration_total",
                                                      "fix_count_posative_acceleration",
                                                      "fix_diff_max_min_acceleration",
                                                      "fix_factor_max_min_acceleration"] \
                                                   + stat_names("fixation_deacceleration_", True) \
                                                   + ["fix_deacceleration_total",
                                                      "fix_count_negative_acceleration",
                                                      "fix_diff_max_min_deacceleration",
                                                      "fix_factor_max_min_deacceleration"]

    return [num_fixation,
            fixation_rate] \
           + fixation_time_stats \
           + [total_fixation_time] \
           + fixation_distance_stats \
           + [total_fixation_distance] \
           + fixation_speed_stats \
           + [total_fixation_speed] \
           + fixation_angular_vel_stats \
           + [total_fixation_angular_vel] \
           + angular_acceleration_feats \
           + angular_deacceleration_feats \
           + acceleration_feats \
           + deacceleration_feats \
        , non_distributional_fixation_features_headers


# def histogram(angle_3points_list, session, participant_id, steps):
def histogram(angle_3points_list, steps):
    """ This function to calculate the histogram features

    Input:
    angle_3points_list
    steps
    session,participant_id

    Output:
    Histogram features
    """
    assert len(angle_3points_list.shape) == 2
    # print(len(angle_3points_list))
    bins = np.arange(0, steps + 1, 1)
    # print(angle_3points_list)
    hist_data, hist_bins = np.histogram(angle_3points_list, bins)
    total = np.sum(hist_data)
    histogram_data = np.append(hist_data, statistics(hist_data))
    histogram_data = np.append(histogram_data, [total])

    header = []
    counter = 1
    for step in range(1, len(hist_data) + 1):
        letter = 'step_' + str(counter)
        header.append(letter)
        counter = counter + 1

    stat = stat_names("histogram_", True)
    stat.extend(['histogra_total'])
    header.extend(stat)

    return histogram_data, header


def micro_fixation(xy, distance_threshold=None, name_prefix=""):
    """ calculate micro fixation

        Input:
        2d gaze points array
        distance threshold

        Output:
        frame of micro_fix counts and their stats
        """
    assert len(xy.shape) == 2
    counter = 1
    results = [1]
    # calculate microfixation without distance threshold
    if distance_threshold is None:
        for i in range(1, len(xy)):
            if xy[i][0] == xy[i - 1][0] and xy[i][1] == xy[i - 1][1]:
                # check the current point with the previous if they are smilar the counter will HAVE THE same val
                counter = counter
            else:
                counter = counter + 1
            results.append(counter)
    #  calculate microfixation with using distance threshold
    else:
        distance_vector = calculate_distance_vector(xy)
        distance_vector = np.insert(distance_vector, 0, 0)
        # insert 0 value at zero index because for fisrt point we dont have distance
        for i in range(1, len(xy)):
            if distance_vector[i - 1] <= distance_threshold:
                # check the distanses if it is less than the threshold to conseder them as microfixation
                counter = counter
            else:
                counter = counter + 1
            results.append(counter)
    micro_fix_arr = np.hstack(list(collections.Counter(results).values()))
    count_micro_fix = [micro_fix_arr.shape[0]]
    count_mic_fix_without1 = np.count_nonzero(micro_fix_arr != 1)
    count_micro_fix.append(count_mic_fix_without1)  # append count_mic_fix_without1
    header = ['{}_count_all_microfix'.format(name_prefix), '{}_count_microfix_without_1'.format(name_prefix)]
    for i in range(1, 21):
        count_micro_fix.append(np.count_nonzero(micro_fix_arr == i))
        header.append('{}_count_microfix_{}'.format(name_prefix, i))
    count_micro_fix.append(np.count_nonzero(micro_fix_arr > 20))
    micro_fix_stats = statistics(np.array(count_micro_fix))
    micro_fix_data = np.concatenate((count_micro_fix, micro_fix_stats), axis=0)
    header.extend(stat_names(name_prefix, True))

    return micro_fix_data, header


def ngram_bins(angle_2points_list, steps):
    """ find the directions

    Input:
    angle_2points_list: lists of angles
    steps number of directions

    Output:
    binned_data list of indices
    """
    # steps = 8
    step = 360 / steps
    bins = np.append(np.array([-180]), np.append(np.asarray(np.arange(-180 + step / 2, 180 + step / 2, step)),
                                                 np.array([180])))  # discretization of unit circle [-180,180]

    # bins = np.append(bins, np.array([1000]))    # for no movement direction

    binned_data = np.digitize(angle_2points_list, bins)

    return binned_data


def ngram_features(binned_data, n, name_prefix=""):
    """ find n-gram  which is a contiguous sequence of n items

    Input:
    session,participant_id
    binned_data
    n : contiguous sequence of the directions which the ngram type(uni, bi, tri ..)

    Output:
    ngram data frame

    """
    count_ang = collections.Counter(zip(*[binned_data[i:] for i in range(n)]))

    base_sequence = [1, 2, 3, 4, 5, 6, 7, 8]  # all possible base directions
    header_data = list(product(set(base_sequence), repeat=n))
    # get all posibile direction list according to the value of n
    total = sum(count_ang.values())

    # header = ["direction_{}".format(i) for i in range(1,len(header_data)+1)]
    header = ["{}_direction_{}".format(name_prefix, i) for i in header_data]  # get all direction in header

    gram_data = [count_ang[step] for step in header_data]  # get the count of each direction
    header.extend(stat_names(name_prefix) + ['{}_total'.format(name_prefix)])
    gram_data = np.concatenate((gram_data, list(statistics(np.array(gram_data))), [total]), axis=0)

    return gram_data, header
