from schau_mir_in_die_augen.process.trajectory import Trajectory, Trajectories

import numpy as np


class EyeMovementClassifier:

    def __init__(self, name: str = None,
                 parameter: dict = None,
                 mark_nan: bool = False,  # mark_outs = False
                 minimum_sample_length: int = 5  # analog to old version
                 ):
        """
        :param name:
        :param parameter:
        :param mark_nan:
        :param minimum_sample_length: other samples will be marked by sample type "Short X"
        """
        self.name = name
        self.parameter = parameter
        self.mark_nan = mark_nan
        self.minimum_sample_length = minimum_sample_length


class EyeMovementClassifiers:

    def __init__(self, eye_movement_classifier: (EyeMovementClassifier, list) = None):

        if isinstance(eye_movement_classifier, EyeMovementClassifier):
            self.subset_classifiers = [eye_movement_classifier]
        elif isinstance(eye_movement_classifier, list):
            if all([isinstance(emc, EyeMovementClassifier) for emc in eye_movement_classifier]):
                self.subset_classifiers = eye_movement_classifier
            else:
                raise Exception('Got a list with not only EyeMovementClassifier!')
        else:
            raise Exception('EyeMovementClassifier could be EyeMovementClassifiers, but not {}'.format(
                type(eye_movement_classifier)))

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.subset_classifiers):
            raise StopIteration
        else:
            self.index += 1
            return self[self.index - 1]

    def __getitem__(self, item):
        return self.subset_classifiers[item]

    def __len__(self):
        return len(self.subset_classifiers)


class EyeMovementClassification:
    """ Split the Trajectory in subsets with tags """

    def __init__(self, trajectories: (Trajectory, Trajectories) = None,
                 subset_classifier: (EyeMovementClassifier, EyeMovementClassifiers) = None):
        """
        :param trajectories: path of eye
        :param subset_classifier: classifier to use
        """
        if isinstance(trajectories, (Trajectory, Trajectories)):
            self._trajectories = Trajectories(trajectories)
        else:
            raise Exception('Expected Trajectory. Got {}'.format(type(trajectories)))

        if isinstance(subset_classifier, EyeMovementClassifier):
            self._subset_classifier = EyeMovementClassifiers(subset_classifier)
        elif isinstance(subset_classifier, EyeMovementClassifiers):
            self._subset_classifier = subset_classifier
        else:
            raise Exception('Expected EyeMovementClassifier. Got {}'.format(type(subset_classifier)))

        self._subset_starts = [[[]] * len(self._trajectories)] * len(self._subset_classifier)
        self._subset_types = [[[]] * len(self._trajectories)] * len(self._subset_classifier)

    @property
    def subset_starts(self):
        return self._subset_starts

    @property
    def subset_types(self):
        return self._subset_types

    def get_ids(self, trajectory_ids=None, classifier_ids=None, only_list=True):
        """ get ids for calculation """

        if trajectory_ids is None:
            trajectory_ids = range(len(self._trajectories))
        elif only_list and isinstance(trajectory_ids, int):
            trajectory_ids = [trajectory_ids]

        if classifier_ids is None:
            classifier_ids = range(len(self._subset_classifier))
        elif only_list and isinstance(classifier_ids, int):
            classifier_ids = [classifier_ids]

        return trajectory_ids, classifier_ids

    def get_subs(self, trajectory_ids=None, classifier_ids=None):

        trajectory_ids, classifier_ids = self.get_ids(trajectory_ids, classifier_ids, only_list=False)

        if isinstance(trajectory_ids, (list, range)) and isinstance(classifier_ids, (list, range)):
            return [[self._subset_starts[t_id][c_id] for t_id in trajectory_ids] for c_id in classifier_ids], \
                   [[self._subset_types[t_id][c_id] for t_id in trajectory_ids] for c_id in classifier_ids]
        elif isinstance(trajectory_ids, int) and isinstance(classifier_ids, int):
            return self._subset_starts[trajectory_ids][classifier_ids], \
                   self._subset_types[trajectory_ids][classifier_ids]
        else:
            raise Exception('Ids should both be either lists/range or ints. Got "{t}" and {c}'.format(
                t=type(trajectory_ids), c=type(classifier_ids)))

    def eye_movement_classification(self, tray_id: int = None, class_id: int = None):

        self._subset_starts[tray_id][class_id], self._subset_types[tray_id][class_id] = \
            eye_movement_classification(self._trajectories[tray_id], self._subset_classifier[class_id])

        return self._subset_starts[tray_id][class_id], self._subset_types[tray_id][class_id]

    def mark_nan_sample(self, tray_id: int = None, class_id: int = None):

        self._subset_starts[tray_id][class_id], self._subset_types[tray_id][class_id] = \
            mark_nan_sample(self._trajectories[tray_id].isnan,
                            self._subset_starts[tray_id][class_id],
                            self._subset_types[tray_id][class_id])

        return self.get_subs(tray_id, class_id)

    def mark_short_subsets(self, tray_id: int = None, class_id: int = None):

        for idd in range(len(self._subset_types[tray_id][class_id])):
            if self._subset_starts[tray_id][class_id][idd+1] - self._subset_starts[tray_id][class_id][idd] \
                    < self._subset_classifier[class_id].minimum_sample_length:
                self._subset_types[tray_id][class_id][idd] = 'short ' + self._subset_types[tray_id][class_id][idd]

        return self.get_subs(tray_id, class_id)

    def do_classification(self, trajectory_id: int = None, classifier_id: int = None):

        trajectory_ids, classifier_ids = self.get_ids(trajectory_id, classifier_id)

        for i_tray in trajectory_ids:
            for i_class in classifier_ids:

                self.eye_movement_classification(i_tray, i_class)

                if self._subset_classifier[i_class].mark_nan:
                    self.mark_nan_sample(i_tray, i_class)

                # if mark_outlier:
                #     mark_out_sample()

                self.mark_short_subsets(i_tray, i_class)

        return self.get_subs(trajectory_id, classifier_id)

    def get_subset(self, kind: str, tray_id: int = None, class_id: int = None, subset_id: int = None):

        if tray_id >= len(self._trajectories):
            raise Exception("There are only {} trajectories. You asked for trajectory {}!".format(
                len(self._trajectories), tray_id))
        if class_id >= len(self._subset_classifier):
            raise Exception("There are only {} subsets. You asked for Subset {}!".format(
                len(self._subset_classifier)-1, class_id))
        if subset_id >= len(self._subset_starts):
            raise Exception("There are only {} subsets. You asked for Subset {}!".format(
                len(self._subset_starts)-1, subset_id))

        return self._trajectories[tray_id].get_trajectory(kind, return_copy=False)[
               self._subset_starts[
                   tray_id][class_id][subset_id]:self._subset_starts[
                   tray_id][class_id][subset_id + 1],
               :]

    def get_type_overview(self, tray_id: int = None, class_id: int = None):
        keys = set(self.subset_types[tray_id][class_id])
        overview = {}

        for key in keys:
            overview[key] = self.subset_types[tray_id][class_id].count(key)

        return overview


def eye_movement_classification(trajectory: Trajectory,
                                eye_movement_classifier: EyeMovementClassifier):
    """ Select function to to classification

    :param trajectory: with information about trajectory and type
    :param eye_movement_classifier: Method Name and Parameter.
    :return: list of tuples

        For every change of type the sample id is given with the type name.
        The last tuple marks the end of the trajectory.
        e.g. [(0,'saccade'),(105,'fixation'),...,(2039,'End')
    """

    if eye_movement_classifier.name == 'IVT_new':
        # todo: usually slower than the old ... could be improved someday ...
        velocities = trajectory.get_velocity('angle_deg')
        eye_movement_classifier.parameter['sample_rate'] = trajectory.sample_rate
        sample_start, sample_type = ivt2(velocities, **eye_movement_classifier.parameter)
    elif eye_movement_classifier.name == 'IVT' \
            or eye_movement_classifier.name == 'IVT_old':
        velocities = trajectory.get_velocity('angle_deg')
        eye_movement_classifier.parameter['sample_rate'] = trajectory.sample_rate
        sample_start, sample_type = transform_ivt(*ivt(velocities, **eye_movement_classifier.parameter))
    else:
        raise Exception('Type "{}" is not implemented yet')

    return sample_start, sample_type


def ivt2(velocities, vel_threshold, min_fix_duration, sample_rate):
    """ Extract saccades and fixations from a list of velocities

    :param velocities:  ndarray
        1D array of velocities
    :param vel_threshold: float
        minimum velocity for a saccade
    :param min_fix_duration:
        fixations shorter than this threshold will be added to the sorrounding saccade
    :param sample_rate: float
        Sensor sample rate in Hz
    :return: list of tuples, list of tuples
        List of saccades and list of fixations. Each List contains tuples that contain start and
        end frame.
    """
    assert len(velocities) > 0, "IVT needs at least one element"

    # internally we work with frames, so convert to frame number
    mdf_frames = min_fix_duration * sample_rate

    # mark every possible fixation frame with true
    fixx = velocities < vel_threshold

    count = None
    # remove short fixxations
    # noinspection PyTypeChecker
    for idd in range(len(fixx)):
        # noinspection PyUnresolvedReferences
        if fixx[idd]:
            if count:
                if count >= mdf_frames:
                    continue
                else:
                    count += 1
            else:
                count = 1
        else:
            if count:
                if count < mdf_frames:
                    fixx[idd-count:idd] = False
                    count = None
                else:
                    count = None
            else:
                continue

    # remove not enough last fixations
    # noinspection PyUnboundLocalVariable
    idd += 1
    if count:
        if count < mdf_frames:
            # noinspection PyUnboundLocalVariable
            fixx[idd - count:idd] = False

    # find all changes from saccade to fixation and the other way around
    # these changes mark the end of each sub sequence
    diffs = fixx[:-1] != fixx[1:]

    # The first element is always a start and the last an end.
    # This expands the length of the marker arrays to the input size again
    start_marks = np.insert(diffs, 0, True)

    # get the indices for each sub-region
    sample_start = list(np.where(start_marks)[0])

    # produce type
    # noinspection PyUnresolvedReferences
    if fixx[0]:
        sample_type = ['saccade' if idd % 2 else 'fixation' for idd in range(len(sample_start))]
    else:
        sample_type = ['fixation' if idd % 2 else 'saccade' for idd in range(len(sample_start))]

    # the last element is the length of the velocities! (not the trajectory)
    sample_start.append(len(velocities))

    return sample_start, sample_type


def ivt(velocities, vel_threshold, min_fix_duration, sample_rate):
    """ Extract saccades and fixations from a list of velocities

    :param velocities:  ndarray
        1D array of velocities
    :param vel_threshold: float
        minimum velocity for a saccade
    :param min_fix_duration:
        fixations shorter than this threshold will be added to the sorrounding saccade
    :param sample_rate: float
        Sensor sample rate in Hz
    :return: list of tuples, list of tuples
        List of saccades and list of fixations. Each List contains tuples that contain start and
        end frame.
    """
    assert len(velocities) > 0, "IVT needs at least one element"

    # internally we work with frames, so convert to frame number
    mdf_frames = min_fix_duration * sample_rate

    # mark every possible fixation frame with true
    fixx = velocities < vel_threshold
    # find all changes from saccade to fixation and the other way around
    # these changes mark the end of each sub sequence
    diffs = fixx[:-1] != fixx[1:]

    # The first element is always a start and the last an end.
    # This expands the length of the marker arrays to the input size again
    start_marks = np.insert(diffs, 0, True)
    end_marks = np.append(diffs, True)

    # get the indices for each sub-region
    starts = np.where(start_marks)[0]
    ends = np.where(end_marks)[0]

    # for each sub-region, is it a fixation?
    # noinspection PyUnresolvedReferences
    is_fix = fixx[starts]
    # filter fixations that are too short and combine them with the surrounding saccades
    frame_durations = (ends - starts) + 1
    # all groups we should delete
    rem_ids = np.where((frame_durations < mdf_frames) & is_fix)[0]

    # mark removed fixations as saccades
    # noinspection PyUnresolvedReferences
    fixx[starts[rem_ids]] = False

    # remove short fixations start and end and further:
    # 1. for every short fixation that is not first, also remove the previous saccade's end
    rem_ids_end = np.concatenate([rem_ids, rem_ids - 1])
    ends = np.delete(ends, rem_ids_end[(rem_ids_end < len(ends) - 1) & (rem_ids_end >= 0)])
    # 2. for every short fixation that is not last, also remove the successor saccade's start
    rem_ids_start = np.concatenate([rem_ids, rem_ids + 1])
    starts = np.delete(starts, rem_ids_start[(rem_ids_start <= len(starts) - 1) & (rem_ids_start > 0)])

    # take every second item as it alternates between fixation and saccade
    saccades = zip(starts[::2], ends[::2])
    fixations = zip(starts[1::2], ends[1::2])
    # noinspection PyUnresolvedReferences
    if fixx[starts[0]]:
        # if the xy sequence starts with a fixation, switch lists
        saccades, fixations = fixations, saccades

    # todo: ivt is returning 1 sample less than exist ...

    return list(saccades), list(fixations)


def transform_ivt(sacc, fixx):
    """ Extract saccades and fixations from a list of velocities

    Idea for new more general Algorithm.
        Status now: Convert IVT (slow)

    :param sacc: [(a,b),(c,d),...]
    :param fixx: [(b,c),(d,e),...] or a=>b and b=>a ...
    :return: [NEW] list of starts, list of types

        List of start frames (change from one type to next).
            Last Element of start Frame is end of LastFrame+1

        List of detected Types
            0 = none
            1 = saccade
            2 = fixation
    """

    # pop will otherwise influence the original list outside
    sacc = sacc.copy()
    fixx = fixx.copy()

    sample_start = []
    sample_type = []

    if not sacc and not fixx:
        return sample_start, sample_type
    elif not sacc and len(fixx) == 1:
        sample_start = list(fixx[0])
        sample_start[-1] += 1
        return sample_start, ['fixation']
    elif not fixx and len(sacc) == 1:
        sample_start = list(sacc[0])
        sample_start[-1] += 1
        return sample_start, ['saccade']
    elif not sacc or not fixx:
        raise Exception("Something's wrong ...")

    # if first element is not a saccade, take it from fixation
    if sacc[0][0] > fixx[0][0]:
        sample_start.append(fixx.pop(0)[0])
        sample_type.append('fixation')

    # if one is longer than the other, spare last element for later
    if len(sacc) > len(fixx):
        add_later = sacc.pop(-1)
        add_later_type = 'saccade'
    elif len(fixx) > len(sacc):
        raise Exception('Saccades and Fixations are expected to be alternately')
    else:
        add_later = False

    # append samples alternating
    # this is very lousy and could be much faster
    for i_element in range(len(sacc)):
        sample_start.append(sacc[i_element][0])
        sample_type.append('saccade')
        sample_start.append(fixx[i_element][0])
        sample_type.append('fixation')

    if add_later:
        # noinspection PyUnresolvedReferences
        sample_start.append(add_later[0])
        # noinspection PyUnboundLocalVariable
        sample_type.append(add_later_type)
        # noinspection PyUnresolvedReferences
        sample_start.append(add_later[1]+1)
    else:
        # noinspection PyUnresolvedReferences
        sample_start.append(fixx[-1][1]+1)

    return sample_start, sample_type


def sliding_window(list_len, window_len, step_size):
    # TODO: add sample rate?
    return zip(range(0, list_len, step_size), range(window_len - 1, list_len, step_size))


def mark_nan_sample(nan_index, sample_start, sample_type):

    if isinstance(nan_index[0], (bool, np.bool_)):
        nan_index = np.where(nan_index)[0]
    elif not isinstance(nan_index[0], int):
        raise Exception('Only Bool or Integers accepted! Got "{}"'.format(type(nan_index[0])))

    max_sample = max(sample_start)

    indi = None
    for ii in range(len(nan_index)):

        if indi is None:
            indi = nan_index[ii]
        if ii+1 < len(nan_index) and nan_index[ii+1] == nan_index[ii] + 1:
            continue
        # Trajectory NaN, affects Velocity NaN before and after him.
        # Movement is between nodes!
        sample_start, sample_type = set_sample_type((max(indi-1, 0),
                                                     min(max_sample, nan_index[ii]+1)),
                                                    'NaN', sample_start, sample_type)
        indi = None

    return sample_start, sample_type


def find_first_greater(list_numbers, value):
    try:
        i = next(i for i, v in enumerate(list_numbers) if v > value)
    except StopIteration:
        i = None
    return i


class SetSampleError(Exception):
    pass


""" Replace one or more sample types """
def set_sample_type(index_range, new_type, sample_start, sample_type, scale_up: bool = False):

    if index_range[1]-index_range[0] < 1:
        raise SetSampleError('Range has to be at least one sample. Got {} to {}'.format(index_range[0], index_range[1]))

    index_in = find_first_greater(sample_start, index_range[0]-1)
    index_out = find_first_greater(sample_start, index_range[1])

    if (index_in is None or index_in == sample_start[-1]) or \
            (index_out is None and not index_range[1] == max(sample_start) and not scale_up):
        raise SetSampleError('Range {i_i} to {i_o} not in Samples {s_i} to {s_o}'.format(
            i_i=index_range[0], i_o=index_range[1], s_i=min(sample_start), s_o=max(sample_start)))
    if index_out is None:
        del sample_start[-1]
        index_out = len(sample_start)

    type_at_end = sample_type[index_out-1]

    # add new_type
    sample_start = sample_start[:index_in] + [index_range[0]] + sample_start[index_in:]
    sample_type = sample_type[:index_in] + [new_type] + sample_type[index_in:]

    index_out += 1

    # continue old_type
    sample_start = sample_start[:index_out] + [index_range[1]] + sample_start[index_out:]
    sample_type = sample_type[:index_out] + [type_at_end] + sample_type[index_out:]

    # remove everything in between

    sample_start = sample_start[:index_in+1] + sample_start[index_out:]
    sample_type = sample_type[:index_in+1] + sample_type[index_out:]

    # remove last element of sample_type, if it was overwritten (index_out last element)
    if len(sample_start) == len(sample_type):
        del sample_type[-1]

    return sample_start, sample_type
