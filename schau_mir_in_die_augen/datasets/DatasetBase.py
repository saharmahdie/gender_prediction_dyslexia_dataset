from abc import ABC, abstractmethod
import logging
import time
import numpy as np

from schau_mir_in_die_augen.process.trajectory import Trajectory, Trajectories
from schau_mir_in_die_augen.process.conversion import convert_pixel_coordinates_to_angles_deg

class DatasetBase(ABC):

    def __init__(self,
                 dataset_name: str = None,
                 sample_rate: float = 0,
                 data_kind: str = None,
                 seed: int = None,
                 user_limit: int = None):

        self.dataset_name = dataset_name    # long name (changable)
        self.dataset_key = None             # short name (fix)

        self.sample_rate = sample_rate
        self.data_kind = data_kind
        self.seed = seed
        self.user_limit = user_limit

        self.info_time = 10  # seconds (how often updates will be print with ETA)

    @abstractmethod
    def get_screen_params(self):
        return {'pix_per_mm': np.asarray([0, 0]),
                'screen_dist_mm': 0,
                'screen_res': np.asarray([0, 0])}

    @abstractmethod
    def get_users(self):
        """ Unique list of users

        :return: list of users
        """
        # necessary for visualization
        return []

    @abstractmethod
    def get_cases(self):
        """ List of recordings per user

        :return: list of recordings per user
        """
        # necessary for visualization
        return []

    @abstractmethod
    def load_data(self, user, case, convert=True):
        """ Get x, y array with ordinates for recording $case from $user

        :param user: str
            user id to load (e.g. 'ID_001')
        :param case: str ('1' or '2' for Bioeye)
            identifier for filename to divide training and testing
        :param convert: boolean (default=True)
            will convert array to Pixel

        :return: np.ndarray
            2D np.arrays with same length and x, y as components

            pixel:
            x should be 0 to width from left to right
            y should be 0 to height from top to bottom

            angle:
            x should be around 0 from left to right?
            y should be around 0 from top to bottom?
        """
        # necessary for visualization
        # This Method should set to pixel or angle
        self.data_kind = ''

        return np.asarray([])

    def load_datas(self, users:list, cases:list, partitions:int = 1):
        """ load multiple data sets"""

        logger = logging.getLogger(__name__)
        logger.info('Loading {ds} in {parts} Partitions.\nUsers: {users}\nCases: {cases}\n'.format(
            ds=type(self), parts=partitions, users=users, cases=cases))

        dfs = []
        ids = []
        time_info = time.time()
        for ii, u in enumerate(users):

            for c in cases:

                logger.debug('Loading case "{case}" for user "{user}".'.format(case=c, user=u))

                df, number_of_nan = self.load_data_no_nan(u, c)

                # split samples into $splits parts
                dfs.extend(np.array_split(df, partitions))
                ids.extend(np.repeat(u, partitions).tolist())

            if time.time() - time_info > self.info_time:
                print('Loading {act} of {total}. Will finish in about {sec:.0f} seconds. ETA: {eta}'.format(
                    act=ii+1, total=len(users), sec=len(users)/(ii+1)*(time.time() - time_info),
                    eta=time.strftime('%H:%M:%S',time.localtime(time.time()
                                                                + len(users)/(ii+1)*(time.time() - time_info)
                                                                ))))
                time_info = time.time()

        return dfs, ids

    # noinspection PyUnusedLocal
    def get_gender(self, user: str):
        """ Convert username in gender (Method to override) """
        return None

    # noinspection PyUnusedLocal
    def get_label(self, user: str):
        """ Convert username in custom label (Method to override) """
        return None

    # noinspection PyUnusedLocal
    def get_age(self, user: str):
        """ Convert username in custom label (Method to override) """
        return None

    # noinspection PyMethodMayBeStatic
    def modify_trajectory(self, trajectory):
        """ Change loaded trajectory depending on Dataset.
        Will be called for every "load_trajectory"
        """
        return trajectory

    def load_trajectory(self, user: str, case: str) -> Trajectory:
        """ Get trajectory and all regarding information in Trajectory class object
        :param user: str
            user id to load (e.g. 'ID_001')
        :param case: str
            identifier for filename to divide training and testing

        :return: Trajectory (see class)
        """
        screen_params = self.get_screen_params()

        # todo: this is really ugly ...
        if self.data_kind != 'pixel_shifted':
            if self.data_kind == 'angle_deg':
                screen_params['screen_res'] = convert_pixel_coordinates_to_angles_deg(
                    xy=screen_params['screen_res'] / 2,
                    pix_per_mm=screen_params['pix_per_mm'],
                    screen_dist_mm=screen_params['screen_dist_mm']) * 2
            else:
                #raise Exception
                pass

        screen_params['fov_x'] = screen_params['screen_res'][0]
        screen_params['fov_y'] = screen_params['screen_res'][1]
        del screen_params['screen_res']
 #### label=self.get_age(user),
        try:
            trajetory = self.modify_trajectory(Trajectory(
                self.load_data(user, case, convert=False),
                self.data_kind,
                user=user, case=case, gender=self.get_gender(user), label_by_dataset=self.get_label(user), age=self.get_age(user),
                sample_rate=self.sample_rate,
                **screen_params))
        except IOError:
            print('Could not load user: "{user}" with case "{case}"!'.format(user=user, case=case))
            raise

        return trajetory

    def load_trajectories(self, users: list, cases: list):
        """ load multiple data sets"""

        logger = logging.getLogger(__name__)
        logger.info('Loading {ds}.\nUsers: {users}\nCases: {cases}\n'.format(
            ds=type(self), users=users, cases=cases))

        trajectories = Trajectories()
        time_info = time.time()
        for ii, u in enumerate(users):

            for c in cases:

                logger.debug('Loading case "{case}" for user "{user}".'.format(case=c, user=u))

                trajectories.append(self.load_trajectory(u, c))

            if time.time() - time_info > self.info_time:
                print('Loading {act} of {total}. Will finish in about {sec:.0f} seconds. ETA: {eta}'.format(
                    act=ii+1, total=len(users), sec=len(users)/(ii+1)*(time.time() - time_info),
                    eta=time.strftime('%H:%M:%S',time.localtime(time.time()
                                                                + len(users) / (ii+1)
                                                                * (time.time() - time_info)
                                                                ))))
                time_info = time.time()

        return trajectories

    @abstractmethod
    def get_cases_training(self) -> list:
        return self.get_cases()

    def load_training(self,partitions=1):
        """ Loads a list of samples

        :param partitions: int
        Split each trajectory into $partitions samples

        self.user_limit: int
        Maximum number of samples per user

        :return: sampleRate: float, dfs: [(np.array(x,y)], id: [int]
        """

        return self.load_datas(self.get_users(), self.get_cases_training(), partitions)

    def load_training_trajectories(self):

        return self.load_trajectories(self.get_users(), self.get_cases_training())

    @abstractmethod
    def get_cases_testing(self) -> list:
        return self.get_cases()

    def load_testing(self, partitions=1):
        """ Loads a list of samples

        :param partitions: int
        Split each trajectory into $partitions samples

        self.user_limit: int
        Maximum number of samples per user

        :return: sampleRate: float, dfs: [(np.array(x,y)], id: [int]
        """

        return self.load_datas(self.get_users(), self.get_cases_testing(), partitions)

    def load_testing_trajectories(self):
        return self.load_trajectories(self.get_users(), self.get_cases_testing())

    @abstractmethod
    def get_stimulus(self, case='S1'):
        """ Relative filename of stimulus

        :param case: case from get_cases
        :return: string
           Relative path to stimulus
        """
        print("This Method was not yet overwritten by child ({}).".format(self.__class__))
        # necessary for visualization
        return ""

    def load_data_no_nan(self, user, case):
        """ Wraps load_data and replaces NaNs by interpolation.
        Returns pixel format
        """
        # todo: very dangerous method. Should have warning and limit of filling.
        df = self.load_data(user, case)  # this always loads pixel
        # dataset contains nans, interpolate linearly
        nans, x = self.nan_helper(df)
        df[nans, 0] = np.interp(x(nans), x(~nans), df[~nans, 0])
        df[nans, 1] = np.interp(x(nans), x(~nans), df[~nans, 1])

        return df, sum(nans).max()

    @staticmethod
    def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.
           From: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

        Input:
            - y, 2d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= DatasetBase.nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """
        assert len(y.shape) == 2
        return np.isnan(y[:,0]+y[:,1]), lambda z: z.nonzero()[0]
