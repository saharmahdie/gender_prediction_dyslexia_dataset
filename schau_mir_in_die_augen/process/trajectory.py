import numpy as np
import matplotlib.pyplot as plt
import copy
import collections

from inspect import getouterframes, currentframe
from os.path import basename

import schau_mir_in_die_augen.process.filtering as filtering


def print_file_and_line():
    print('Message from:')
    frame_infos = getouterframes(currentframe())[3:]
    for frame_info in frame_infos:
        print(' "{function}" ({file}:{line})'.format(
            function=frame_info.function,
            file=basename(frame_info.filename),
            line=frame_info.lineno))

def calculate_derivation(xy: np.ndarray, sample_rate: float) -> np.ndarray:
    """ Calculate x and y derivation for a trajectory
    if xy is position it returns velocity. if xy is velocity it returns acceleration
    Returns positive as well as negative values.

    Input:
    :param xy: 1D or 2D trajectory
    :param sample_rate: Sample rate in Hz

    :return: 1D or 2D array of derivation, len = len(v) - 1
        unit * 1/s => unit/s

    """
    if len(xy.shape) > 1:  # does this work for other than 2?
        return (xy[1:, :] - xy[:-1, :]) * sample_rate
    elif len(xy.shape) == 1:
        return (xy[1:] - xy[:-1]) * sample_rate


class TrajectoryBase:
    """ This class should store an eye-movement-trajectory xy.
    Depending on more parameters you can modify the trajectory.

    See initialize function for more information.

    xy is either in pixel, in angle_deg or in angle_rad
    The mean value or center of image should be (0,0).

    """

    def __init__(self, xy):
        """ Initialization of Trajecotry

        @:param xy: [x,y] 2D np.ndarray or list of two lists
            Trajectory data
        """

        # copy() makes shure we have independent data, wich can not been acessed from the outside.

        if isinstance(xy,np.ndarray):
            self._xy = copy.deepcopy(xy)
        elif isinstance(xy,list) and len(xy[0]) == 2:
            self._xy = np.asarray(copy.deepcopy(xy))
        else:  # todo: add more opportunities
            raise Exception('xy should be a ndarray. Got "{}"'.format(type(xy)))
        if self._xy.shape[1] != 2:
            raise Exception('xy should have 2 columns. Got "{}"'.format(self._xy.shape[1]))

    def __len__(self):
        return self._xy.shape[0]

    def copy(self):  # is this really necessary?
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __eq__(self, other):
        if type(self) == type(other):
            return self.__dict__ == other.__dict__
        return False

    @property
    def xy(self):
        return self._xy

    @property
    def sample_id(self):
        return list(range(self.xy.shape[0]))

    @property
    def statistics(self):
        return {'cog':np.nanmean(self._xy,axis=0),
                'std':np.nanstd(self._xy,axis=0)}

    def plot(self):
        plt.plot(self._xy[:, 0], label='x')
        plt.plot(self._xy[:, 1], label='y')
        plt.legend()
        plt.ylabel('trajectory value')
        plt.show()


class TrajectoryKind(TrajectoryBase):
    """ Provides the kind of the Trajectory (angle or pixel) """

    def __init__(self, xy, kind:str):
        """
        @:param kind: str
            type of the data (pixel or angle)
        """
        super().__init__(xy)

        if not kind:
            print('W: There was no kind given for dataset')
        elif not isinstance(kind, str):
            raise Exception('Kind should be a string. Got "{}"'.format(type(kind)))

        self._kind = None
        self._set_kind(kind)

    def _get_kind(self):
        return self._kind

    def _set_kind(self, kind):
        self._kind = kind

    kind = property(_get_kind,_set_kind)


class TrajectoryProcess(TrajectoryKind):
    """ Provides documentation for Changes

    ._process: str
        will store a list of all changes to data
    """

    def __init__(self, xy, kind: str):

        self._processing = []
        super().__init__(xy, kind)
        self._add_process('initialize', 'initialize', self._kind)

    @property
    def processing(self):
        return self._processing

    def _add_process(self,process_type:str, process_name:str,process_parameter=None):
        """ This function is to document the changes to the data in variable "_processing"

        @:param process_type: string
            intern keyword
        @:param process_name: string
            description of change
        @:param process_parameter: any
            value of change
        """

        if process_type in ['filter', 'conversion', 'interpolation', 'removal', 'extension', 'data', 'calculation']:
            # reset velocities
            self._velocity = None
        elif process_type not in ['initialize', 'offset']:
            raise Exception("Process unknown! Bad Programmer")
        self._processing.append((process_name, process_parameter))
        return


class TrajectoryModify(TrajectoryProcess):
    """ Methods to change the content of Trajectory """

    def __init__(self, xy, kind: str):

        super().__init__(xy, kind)

        self._interpolated = [False] * self._xy.shape[0]
        self._removed_duplicates = [False] * self._xy.shape[0]

    @property
    def isnan(self):
        return np.isnan(self._xy[:, 0]+self._xy[:, 1])

    def apply_filter(self, filter_type: str, **kwargs):
        """ Use some Filter on data.

        :param filter_type: name of Filter (e.g. "savgol")
        :param kwargs: Parameter for selected Filter
        """
        if filter_type == 'savgol':
            self._xy = filtering.savgol_filter_trajectory(self._xy, **kwargs)
            self._add_process('filter', filter_type, kwargs)
        else:
            raise Exception('"{}" is not yet implemented'.format(filter_type))

    @property
    def interpolated(self):
        return self._interpolated

    def interpolate(self):
        nan_before = self.isnan
        self._xy[self.isnan, 0] = np.interp(np.nonzero(self.isnan)[0],
                                            np.nonzero(~self.isnan)[0],
                                            self._xy[~self.isnan, 0])
        self._xy[self.isnan, 1] = np.interp(np.nonzero(self.isnan)[0],
                                            np.nonzero(~self.isnan)[0],
                                            self._xy[~self.isnan, 1])
        self._interpolated = self.isnan != nan_before
        self._add_process('interpolation', 'interpolate', sum(self._interpolated))
        return self._interpolated

    def remove_duplicates(self, threshold: int = 0):
        """ remove unchanged values

         :param threshold: amount of ignored values (0 = every duplicate, 1 = three ore more, 2 = ..."""
        xy, remove_stats = filtering.remove_duplicates(self._xy, threshold)
        self._removed_duplicates = self.xy != xy
        self._xy = xy
        self._add_process('removal', 'removed duplicates', remove_stats)
        return remove_stats

    def invert_y(self):
        """ y = - y """
        self._xy[:, 1] = - self._xy[:, 1]
        self._add_process('data', 'invert','y')
        return

    def offset(self,offset_x, offset_y):
        """ xy = xy + Offset """
        self._xy[:, 0] = self._xy[:, 0] + offset_x
        self._xy[:, 1] = self._xy[:, 1] + offset_y
        self._add_process('offset', 'offset', (offset_x, offset_y))
        return

    def center(self):
        """ centering data"""
        center_of_gravity = self.statistics['cog']
        self._xy[:, 0] = self._xy[:, 0] - center_of_gravity[0]
        self._xy[:, 1] = self._xy[:, 1] - center_of_gravity[1]
        self._add_process('offset', 'centering', -center_of_gravity)
        return

    def scale(self,scaling_factor,scaling_y:float = None):

        if scaling_y is None:
            self._xy = self._xy * scaling_factor
            self._add_process('calculation', 'scaling', scaling_factor)
        else:
            self._xy[:, 0] = self._xy[:, 0] * scaling_factor
            self._xy[:, 1] = self._xy[:, 1] * scaling_y
            self._add_process('calculation', 'scaling', (scaling_factor, scaling_y))
        return


class TrajectoryFOV(TrajectoryModify):
    """ Optional field of view:

    @:param fov_x: width or radius (y=None) of field of view
    @:param fov_y: float height of field of view
    @:param fov_type: type of field of view - string [default_x='circle',tbd='ellipse',default_xy='rectangle']
    """
    def __init__(self, xy, kind: str,
                 fov_x: float = None, fov_y: float = None, fov_type: str = None):

        if isinstance(fov_x,np.ndarray):
            fov_x = fov_x.item()
        if isinstance(fov_y, np.ndarray):
            fov_y = fov_y.item()

        self._fov_x = fov_x
        self._fov_y = fov_y

        super().__init__(xy, kind)

        if fov_type is None:
            if fov_y is None:
                self._fov_type = 'circle'
            else:
                self._fov_type = 'rectangle'
        else:
            self._fov_type = fov_type

    def __check_fov(self):
        if self._fov_x is None:
            raise Exception('fov_x is not defined')
        if self._fov_type == 'rectangle' and self._fov_y is None:
            raise Exception('fov_y is not defined')

    def _set_kind(self, kind):
        if kind in ['pixel_image', 'pixel_shifted']:
            # data kind has to be centered, so the negative offset will be applied.
            if self._fov_x is None or self._fov_y is None:
                raise Exception('For "pixel_image" or "pixel_shifted" fov information is mandatory!')
            self.offset(- self._fov_x/2, - self._fov_y/2)
            if kind == 'pixel_image':
                # y is inverted and and has to be fixed
                self.invert_y()
            # noinspection PyUnresolvedReferences
            self._TrajectoryConvert__set_kind('pixel')
        else:
            # noinspection PyUnresolvedReferences
            self._TrajectoryConvert__set_kind(kind)

    @property
    def isfov(self):
        self.__check_fov()
        if self._fov_type == 'circle':
            return np.sum(np.square(self._xy),axis=1) <= self._fov_x
        elif self._fov_type == 'rectangle':
            return abs(self._xy[:, 0]) <= self._fov_x and abs(self._xy[:, 0]) <= self._fov_y
        else:
            raise Exception('"{}" is not yet implemented.'.format(self._fov_type))

    def has_same_infos(self, other):
        assert isinstance(other, TrajectoryFOV)
        if self._fov_x != other._fov_x or self._fov_y != other._fov_y:
            return False
        else:
            return True


class TrajectoryConvertError(Exception):

    @staticmethod
    def bad_input(name, value):
        print('E: Input: {}'.format(value))
        return TrajectoryConvertError('"{name}" got unknown input "{type}" or bad shape.'.format(
            name=name, type=type(value)))

    @staticmethod
    def assert_1d(name, value, length:int = None):
        if isinstance(value, (list, tuple)):
            if length is None or len(value) == length:
                value = np.asarray(value)
            else:
                raise TrajectoryConvertError(
                    '"{name}" expected {length} 1D input, but got {number} elements".'.format(
                        name=name, length=length, number=len(value)))
        if isinstance(value, np.ndarray):
            if len(value.shape) != 1:
                raise TrajectoryConvertError(
                    '"{name}" expected 1D input, but got {number} dimensions.'.format(
                        name=name, length=length, number=len(value.shape)))
            elif length is not None and value.shape[0] != length:
                raise TrajectoryConvertError(
                    '"{name}" expected {length} 1D input, but got {number} dimensions.'.format(
                        name=name, length=length, number=value.shape[0]))
        return value

    @staticmethod
    def assert_2d(name, value):
        if isinstance(value, (list, tuple)):
            if len(value) == 2:
                value = np.asarray([value])
            else:
                value = np.asarray(value)
        elif isinstance(value,np.ndarray):
            if len(value.shape) == 1:
                value = np.asarray([value])
        elif value is None:
            return np.ndarray([0,2])
        else:
            raise TrajectoryConvertError('"{name}" expected 2D input, but got bad type: "{type}".'.format(
                name=name, type=type(value)))
        if not value.shape[1] == 2:
            raise TrajectoryConvertError('"{name}" expected 2D input, but got bad shape: "{shape}".'.format(
                name=name, shape=value.shape))
        return value


class TrajectoryConvert(TrajectoryFOV):
    """ Conversion between pixel and angle """

    def __init__(self, xy, kind: str,
                 pix_per_mm=None, screen_dist_mm=None,
                 fov_x=None,fov_y=None,fov_type=None):
        """
        Optional experiment data (needed for conversion):
            @:param pix_per_mm: [float, float]...
                relative resolution of screen
            @:param screen_dist_mm: float
                distance of eyes to the screen
        """

        # experiment_data
        self._pix_per_mm = None
        self.set_pix_per_mm(pix_per_mm)

        self._screen_dist_mm = None
        self.set_screen_dist_mm(screen_dist_mm)

        super().__init__(xy, kind,
                         fov_x=fov_x,fov_y=fov_y,fov_type=fov_type)

    def set_pix_per_mm(self, value):
        value = TrajectoryConvertError.assert_1d('pixel2rad', value)
        self._pix_per_mm = value

    def get_pix_per_mm(self):
        return self._pix_per_mm

    pix_per_mm = property(get_pix_per_mm, set_pix_per_mm)

    def set_screen_dist_mm(self, value):
        if value is None or isinstance(value, (float, int)):
            self._screen_dist_mm = value
        else:
            raise TrajectoryConvertError.bad_input('screen_dist_mm', value)

    def get_screen_dist_mm(self):
        return self._screen_dist_mm

    screen_dist_mm = property(get_screen_dist_mm, set_screen_dist_mm)

    def has_same_infos(self, other):
        assert isinstance(other, TrajectoryConvert)
        if not super().has_same_infos(other) \
                or self._pix_per_mm[0] != other._pix_per_mm[0] \
                or self._pix_per_mm[1] != other._pix_per_mm[1] \
                or self._screen_dist_mm != other._screen_dist_mm:
            return False
        else:
            return True

    @property
    def screen_params(self):
        """ for compatibility """
        return {'pix_per_mm':self._pix_per_mm,
                'screen_dist_mm':self.screen_dist_mm,
                'screen_res':self.get_fov('pixel')}

    @staticmethod
    def screen_params_converter(screen_params: dict):
        screen_params = screen_params.copy()
        screen_params['fov_x'] = screen_params['screen_res'][0]
        screen_params['fov_y'] = screen_params['screen_res'][1]
        del screen_params['screen_res']
        return screen_params

    def __set_kind(self, kind):
        if self._kind is None:
            if kind in ['pixel', 'angle_deg', 'angle_rad']:
                self._kind = kind
            elif kind == 'angle':
                print('W: by "angle" you mean degrees. Use "angle_deg" to suppress this warning.')
                print_file_and_line()
                self._kind = 'angle_deg'
            else:
                raise Exception('"{}" is an unknown Type'.format(kind))
        else:
            self.convert_to(kind)

    @property
    def is_experiment_data(self):
        if self._pix_per_mm is None or self._screen_dist_mm is None:
            return False
        else:
            return True

    def convert_to(self, kind):
        self.get_trajectory(kind, inplace=True, return_copy=False)
        return

    def pixel2rad(self, pixel):
        assert self.is_experiment_data
        pixel = TrajectoryConvertError.assert_2d('pixel2rad', pixel)
        return np.arctan2(pixel, self._screen_dist_mm * self._pix_per_mm)

    def rad2pixel(self, rad):
        assert self.is_experiment_data
        rad = TrajectoryConvertError.assert_2d('pixel2rad', rad)
        return self._pix_per_mm * np.tan(rad) * self._screen_dist_mm

    def get_trajectory(self, kind, inplace: bool = False, data_slice: slice = None, return_copy: bool = True):
        """ Return xy array of type "kind"

        :param kind: 'pixel', 'angle_rad', ... kind of data type
            'pixel_image', 'pixel_shifted' have a special role, they will always be calculated.
            Both have a shift to the first quadrant. 'pixel_image' will invert the y component
        :param inplace: if True the saved data kind will be changed (faster for the next call)
        :param data_slice: return only a part of the trajectory
        :param return_copy: will return a copy of the original data by default
        :return: ndarray
        """
        if data_slice is not None and inplace:
            raise Exception('Can not convert while slicing.')
        if kind == self.kind:
            if data_slice is None:
                xy = self._xy
            else:
                xy = self._xy[data_slice, :]
            if return_copy:
                return copy.deepcopy(xy)
            else:
                return xy

        # it is not relevant for rad to deg
        # if not self.is_experiment_data:
        #     raise Exception('Experiment Information are necessary for conversion!')

        def nope(self_kind, new_kind):
            return Exception('Type "{now}" can not be turned to "{to}" yet!'.format(
                now=self_kind, to=new_kind))

        if kind == 'pixel':
            if self.kind in ['angle_rad', 'angle_deg']:
                xy = self.rad2pixel(self.get_trajectory('angle_rad', return_copy=False, data_slice=data_slice))
            else:
                raise nope(self._kind, kind)
        elif kind in ['pixel_image', 'pixel_shifted']:
            """ move the trajectory out of the center in quadrant 1
                if pixel_image is selected: invert y component bevore.
            """
            if inplace:
                raise Exception('"{}" can\'t be the saved value.'.format(kind))
            xy = self.get_trajectory('pixel', data_slice=data_slice)
            fov = self.get_fov('pixel')
            # can not use += for following lines, because xy could be integer. An Error would occur.
            xy[:, 0] = xy[:, 0] + fov[0] / 2
            if kind == 'pixel_image':
                xy[:, 1] = -xy[:, 1] + fov[1] / 2
            else:
                xy[:, 1] = xy[:, 1] + fov[1] / 2
        elif kind == 'angle_deg':
            if self.kind in ['pixel', 'angle_rad']:
                xy = np.degrees(self.get_trajectory('angle_rad', return_copy=False, data_slice=data_slice))
            else:
                raise nope(self._kind, kind)
        elif kind == 'angle_rad':

            if data_slice is None:
                xy = copy.deepcopy(self._xy)
            else:
                xy = self._xy[data_slice, :]

            if self.kind == 'pixel':
                xy = self.pixel2rad(xy)
            elif self.kind == 'angle_deg':
                xy = np.radians(xy)
            else:
                raise nope(self._kind, kind)
        elif kind == 'angle':
            print('W: by "angle" you mean degrees. Use "angle_deg" to suppress this warning.')
            print_file_and_line()
            kind = 'angle_deg'
            xy = self.get_trajectory(kind, return_copy=False, data_slice=data_slice)
        else:
            raise Exception('"{}" is an unknown type.'.format(kind))
        if inplace:
            self._xy = xy
            self.__convert_fov_to(kind)
            self._add_process('conversion', self._kind, kind)
            self._kind = kind

        if return_copy and data_slice is None:
            return copy.deepcopy(xy)
        else:
            return xy

    def __convert_fov_to(self, kind):
        self.get_fov(kind, inplace=True)

    def get_fov(self, kind, inplace=False):
        if kind == self.kind:
            return np.asarray([self._fov_x, self._fov_y])

        def convert(fun):
            xy = fun(np.asarray([self._fov_x, self._fov_y]) / 2) * 2
            fov_x = xy[0, 0]
            fov_y = xy[0, 1]
            if inplace:
                self._fov_x = fov_x
                self._fov_y = fov_y
            return np.asarray([fov_x, fov_y])

        if kind == 'pixel':
            if self.kind == 'angle_rad':
                return convert(self.rad2pixel)
            elif self.kind == 'angle_deg':
                return convert(lambda x: self.rad2pixel(np.deg2rad(x)))
            else:
                raise Exception
        elif kind == 'angle_deg':
            if self.kind == 'pixel':
                return convert(lambda x: np.rad2deg(self.pixel2rad(x)))
            elif self.kind == 'angle_rad':
                return convert(np.rad2deg)
            else:
                raise Exception
        elif kind == 'angle_rad':
            if self.kind == 'pixel':
                return convert(self.pixel2rad)
            elif self.kind == 'angle_deg':
                return convert(np.deg2rad)
            else:
                raise Exception
        else:
            raise Exception


class TrajectoryCalculate(TrajectoryConvert):
    """ Simple Calculation Methods for Trajectory """

    def __int__(self, xy, kind: str):
        super().__init__(xy, kind)

    def __mul__(self, other: (int, float, list, np.ndarray)):
        result = self.deepcopy()
        result *= other
        return result

    def __imul__(self, other: (int, float, list, np.ndarray)):
        if isinstance(other, (int, float)):
            self._xy *= other
        elif isinstance(other, list):
            if len(other) == 2:
                self._xy[:, 0] *= other[0]
                self._xy[:, 1] *= other[1]
            else:
                raise Exception('List has to have length 2. Got length {}.'.format(len(other)))
        elif isinstance(other, np.ndarray):
            if len(other.shape) == 1 and other.shape[0] == 2:
                self._xy[:, 0] *= other[0]
                self._xy[:, 1] *= other[1]
            else:
                raise Exception('ndarray has to have length 2. Got shape {}.'.format(other.shape))

        else:
            raise Exception('"{}" is an unknown Type for multiplication.'.format(type(other)))
        self._add_process('calculation', 'multiplication', other)
        return self

    def __truediv__(self, other: (int, float)):
        result = self.deepcopy()
        result /= other
        return result

    def __itruediv__(self, other: (int, float)):
        assert isinstance(other, (int, float))
        self._xy /= other
        self._add_process('calculation', 'division', other)
        return self

    def __add__(self, other: (int, float, TrajectoryConvert)):
        result = self.deepcopy()
        result += other
        return result

    def __iadd__(self, other: (int, float, TrajectoryConvert)):
        if isinstance(other, (int, float)):
            self._xy += other
            self._add_process('calculation', 'addition', other)
        elif isinstance(other, TrajectoryConvert):
            self._xy += other.get_trajectory(self.kind)
            self._add_process('calculation', 'addition', 'Trajectory')
        return self

    def extend(self, other: TrajectoryConvert, ignore_infos=False):
        assert isinstance(other, TrajectoryConvert)
        if not ignore_infos and not self.has_same_infos(other):
            raise Exception("Trajectory Infos are not the same!")
        self._xy = np.concatenate([self.xy, other.get_trajectory(self.kind)])
        self._interpolated += other.interpolated
        self._removed_duplicates += other._removed_duplicates
        self._add_process('extension', 'extend', 'Trajectory')
        return self


class TrajectoryVelocity(TrajectoryCalculate):
    """ Provides velocity values to Trajectory"""

    def __init__(self, xy, kind: str, sample_rate: float = 1,
                 pix_per_mm=None, screen_dist_mm=None,
                 fov_x=None, fov_y=None, fov_type=None):
        """
        @:param sample_rate: int
            samples per second
            necessary to calculate a correct velocity in kind per second
        """

        self.sample_rate = sample_rate

        self._velocity = None  # todo: Maybe this is not useful

        super().__init__(xy=xy, kind=kind,
                         pix_per_mm=pix_per_mm, screen_dist_mm=screen_dist_mm,
                         fov_x=fov_x,fov_y=fov_y,fov_type=fov_type)

    @property
    def velocity(self):
        return self.get_velocity()

    def check_sample_rate(self):
        if self.sample_rate is None:
            raise Exception('"sample_rate" is necessary for calculation!')

    def get_velocity(self, kind=None) -> np.ndarray:
        """ Return velocity in actual units or for given units """
        if not kind or kind == self.kind:
            if self._velocity is None:
                self.check_sample_rate()
                vel_xy = calculate_derivation(self._xy, self.sample_rate)
                self._velocity = np.linalg.norm(vel_xy, axis=1)
            return self._velocity
        else:
            vel_xy = calculate_derivation(self.get_trajectory(kind), self.sample_rate)
            return np.linalg.norm(vel_xy, axis=1)

    def plot_velocity(self, kind=None):

        velocity = self.get_velocity(kind)

        plt.plot(velocity, label='velocity '+self.kind)
        plt.legend()
        plt.ylabel(self.kind + '/ s')
        plt.show()

    def has_same_infos(self, other):
        assert isinstance(other, TrajectoryVelocity)
        if not super().has_same_infos(other) \
                or self.sample_rate != other.sample_rate:
            return False
        else:
            return True


class TrajectoryMore:
    """ Provides additional attributes to store content

    .user: user to whom the trajectory belongs
    .case: name of experiment the user made
    .sample_id: basically only one number for each sample (list 0 to n-1)
    """

    def __init__(self, user: str = None, case: str = None, gender: str = None, label_by_dataset: str = None, age: str = None):
        """ Optional Parameters:
        @:param user: name of eye owner
        @:param case: title of test case
        @:param gender: sexuality of eye owner
        @:param age: age group of eye owner
        """

        assert isinstance(user, collections.Hashable)
        assert isinstance(case, collections.Hashable)
        assert isinstance(gender, collections.Hashable)
        assert isinstance(label_by_dataset, collections.Hashable)
        assert isinstance(age, collections.Hashable)

        self.user = user
        self.case = case
        self.gender = gender
        self.label_by_dataset = label_by_dataset
        self.age = age

    def has_same_infos(self, other):
        """ Support Function for Comparison - maybe there are better ways """
        assert isinstance(other, TrajectoryMore)
        if self.user != other.case \
                or self.user != other.case \
                or self.gender != other.gender \
                or self.label_by_dataset != other.label_by_dataset \
                or self.age != other.age:

            return False
        else:
            return True


class Trajectory(TrajectoryVelocity, TrajectoryMore):

    def __init__(self, xy, kind: str, sample_rate: float = 1,
                 pix_per_mm=None, screen_dist_mm=None,
                 fov_x: float = None, fov_y: float = None, fov_type: str = None,
                 user: str = None, case: str = None, gender: str = None, label_by_dataset: str = None, age: str = None):
        # todo: this multiple defining looks not very good
        TrajectoryVelocity.__init__(self, xy=xy, kind=kind,
                                    pix_per_mm=pix_per_mm,          # TrajectoryConvert
                                    screen_dist_mm=screen_dist_mm,  # TrajectoryConvert
                                    sample_rate=sample_rate,        # TrajectoryVelocity
                                    fov_x=fov_x, fov_y=fov_y, fov_type=fov_type)
        TrajectoryMore.__init__(self, user=user, case=case, gender=gender, label_by_dataset=label_by_dataset, age=age)


class TrajectoriesBase:

    def __init__(self, trajectories:(Trajectory, list) = None):
        self.index = 0

        if isinstance(trajectories, Trajectory):
            self._trajectories = [trajectories]
        elif isinstance(trajectories, list):
            if all([isinstance(trajectory, Trajectory) for trajectory in trajectories]):
                self._trajectories = trajectories
            elif len(trajectories) == 0:
                self._trajectories = []
            else:
                raise Exception('Got list with content not of "Trajectory" Class')
        elif trajectories is None:
            self._trajectories = []
        else:
            raise Exception('Got "" not "Trajectory" Class'.format(type(trajectories)))

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        else:
            self.index += 1
            return self[self.index - 1]

    def __getitem__(self, item):
        return self._trajectories[item]

    def __setitem__(self, item, value):
        assert isinstance(value, Trajectory)
        self._trajectories[item] = value

    def __len__(self):
        return len(self.xys)

    def __eq__(self, other):
        if type(self) == type(other):
            return self.__dict__ == other.__dict__
        return False

    @property
    def xys(self):
        return [trajectory.xy for trajectory in self._trajectories]

    def get_trajectories(self, kind: str = None):
        return [trajectory.get_trajectory(kind) for trajectory in self._trajectories]

    @property
    def velocities(self):
        return [trajectory.velocity for trajectory in self._trajectories]

    def get_velocities(self, kind: str = None):
        return [trajectory.get_velocity(kind) for trajectory in self._trajectories]

    @property
    def sample_ids(self):
        return [trajectory.sample_id for trajectory in self._trajectories]

    @property
    def cases(self):
        return [trajectory.case for trajectory in self._trajectories]

    @property
    def users(self):
        return [trajectory.user for trajectory in self._trajectories]

    @property
    def genders(self):
        return [trajectory.gender for trajectory in self._trajectories]

    @property
    def label_by_datasets(self):  # name is generated with "s" at the and.... not really good.
        return [trajectory.label_by_dataset for trajectory in self._trajectories]

    @property
    def ages(self):
        return [trajectory.age for trajectory in self._trajectories]

    @property
    def processing(self):
        """ Return processing of all Trajectories (list).
        Will return list, if processing is not the same.

        :return: list or string
        """
        processing = self._trajectories[0].processing
        identical_bool = [processing == trajectory.processing for trajectory in self._trajectories[1:]]
        if all(identical_bool):
            return processing
        else:
            return 'Processing is different for Trajectories. First difference with index {}.'.format(
                identical_bool.index(False))

    def append(self, trajectory: (Trajectory, list)):
        if isinstance(trajectory, Trajectory):
            self._trajectories.append(trajectory)
        elif isinstance(trajectory, list):
            for tra in trajectory:
                self._trajectories.append(tra)
        else:
            raise Exception('Can not append type "{}" to trajectories'.format(type(trajectory)))
        return

    def select_users(self, user_list: list):
        remove_ids = []
        for idd, trajectory in enumerate(self._trajectories):
            if trajectory.user not in user_list:
                remove_ids.append(idd)
        self.drop(remove_ids)

    def drop(self, indexes: (list, int)):
        if isinstance(indexes, int):
            del self._trajectories[indexes]
        elif isinstance(indexes, list):
            for index in sorted(indexes, reverse=True):
                del self._trajectories[index]
        else:
            raise Exception('Only list or int accepted. Got {}!'.format(type(indexes)))

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def convert_to(self, kind: 'str'):
        [trajectory.convert_to(kind=kind) for trajectory in self]

    def apply_filter(self, filter_type: str, **kwargs):
        [trajectory.apply_filter(filter_type=filter_type, **kwargs) for trajectory in self]

class TrajectoriesCalculation(TrajectoriesBase):

    def __init__(self, trajectories: (Trajectory, list) = None):
        super().__init__(trajectories=trajectories)

    def __add__(self, other: (Trajectory, TrajectoriesBase)):
        result = self.deepcopy()
        if isinstance(other, Trajectory):
            for ii in range(len(self)):
                result[ii] += other
        elif isinstance(other, TrajectoriesBase):
            if len(self) != len(other):
                raise Exception('Can only add Trajectories with the same size!')
            for ii in range(len(self)):
                result[ii] += other[ii]
        else:
            raise Exception('Type "{}" is unknown for addition.'.format(type(other)))
        return result

    def __mul__(self, other: (int, float)):
        assert isinstance(other, (int, float))
        result = self.deepcopy()
        result *= other
        return result

    def __imul__(self, other: (int, float)):
        assert isinstance(other, (int, float))
        for trajectory in self:
            trajectory *= other
        return self

    def __truediv__(self, other: (int, float)):
        assert isinstance(other, (int, float))
        result = self.deepcopy()
        result /= other
        return result

    def __itruediv__(self, other: (int, float)):
        assert isinstance(other, (int, float))
        for trajectory in self:
            trajectory /= other
        return self

    def extend(self, other: (Trajectory, TrajectoriesBase), ignore_infos=False):
        """ Extend the xy coordinates of all Trajectories.
        :param other: Trajectory/Trajectories to add
        :param ignore_infos: don't check for matching infos"""

        if isinstance(other, Trajectory):
            for ii in range(len(self)):
                self[ii].extend(other, ignore_infos)
        elif isinstance(other, TrajectoriesBase):
            if len(self) != len(other):
                raise Exception('Can only extend Trajectories with the same size!')
            for ii in range(len(self)):
                self[ii].extend(other[ii], ignore_infos)
        else:
            raise Exception('Type "{}" is unknown for extension.'.format(type(other)))
        return self

class Trajectories(TrajectoriesCalculation):

    def __init__(self, trajectories: (Trajectory, list) = None):
        super().__init__(trajectories=trajectories)
