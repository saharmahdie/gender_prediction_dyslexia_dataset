# Dyslexia data has 185 users: 111 users 20 sec, 1 user 17.5 sec, 57 users 15 sec and 7 users 10 sec

import numpy as np
from os.path import join as pjoin
import os
import random
from schau_mir_in_die_augen.process.conversion import convert_angles_deg_to_shifted_pixel_coordinates
from schau_mir_in_die_augen.datasets.Dyslexia_base import DyslexiaBase


class Dyslexia(DyslexiaBase):

    def __init__(self, user_limit=None, seed=42, training_data=0.8, use_eye_config='LEFT', load_record_duration=0,
                 male_ratio=None, user_specific_numbers=[0, 0, 0, 0]):
        """

        :param user_limit: int
        to choose the number of users to load
        :param use_eye_config: string
        used to load five eyes data possibilities: left eye data, right eye data, average gaze points of both eyes,
        combine features of both eyes and average features of both eyes
        :param training_data: float
        can be used to define the ratio of data to be kept in training (remaining will go to testing)

        """
        super().__init__(user_limit=user_limit, seed=seed, training_data=training_data, male_ratio=male_ratio, user_specific_numbers=user_specific_numbers)

        self.use_eye_config = use_eye_config
        self.load_record_duration = load_record_duration

    def get_users(self):
        """ return users with same ratio of female to male as in dataset"""
        # unique users
        random.seed(self.seed)
        male_users = sorted([u for u in os.listdir(self.data_folder)
                             if os.path.isdir(pjoin(self.data_folder, u)) and u.endswith(('1', '3'))])   ## the healthy males 3 or disabled 1
        female_users = sorted([u for u in os.listdir(self.data_folder)
                               if os.path.isdir(pjoin(self.data_folder, u)) and u.endswith(('2', '4'))])  ## the healthy females 4 or disabled 2

        # print([u for u in os.listdir(self.data_folder) if os.path.isdir(pjoin(self.data_folder,u))])
        if self.user_specific_numbers != [0, 0, 0, 0]:

            # maybe not necessary, but definite not harmfull
            random.shuffle(male_users)
            random.shuffle(female_users)

            # 1 Male Dyslexic, 2 Female Dyslexic, 3 Male Healthy, 4 Female Healthy
            male_users = random.sample([u for u in male_users if u.endswith('1')], self.user_specific_numbers[0]) + \
                         random.sample([u for u in male_users if u.endswith('3')], self.user_specific_numbers[2])
            female_users = random.sample([u for u in female_users if u.endswith('2')], self.user_specific_numbers[1]) + \
                           random.sample([u for u in female_users if u.endswith('4')], self.user_specific_numbers[3])

            users = male_users + female_users
            self.user_limit = sum(self.user_specific_numbers)

        elif self.user_limit is None:
            if self.male_ratio is not None \
                    and self.male_ratio != len(male_users) / (len(male_users) + len(female_users)):
                print('W: Ignoring male_ratio (all users are selected).')
            users = male_users + female_users
        else:
            # random.seed(self.seed)  # if we want to have random accuracy each time we run the same seed
            # todo: random selection should do the same
            if self.male_ratio is not None:
                female_ratio = 1 - self.male_ratio
            else:
                female_ratio = len(female_users) / (len(male_users) + len(female_users))

            female_n = int(np.round(self.user_limit * female_ratio))

            if female_n > len(female_users):
                print('W: Not enough female users for male ratio {ratio}.'
                      '({need} necessary, {available} available. Males will fill the gap!'.format(
                    ratio=1-female_ratio, need=female_n, available=len(female_users)))
                female_n = len(female_users)

            male_n = self.user_limit - female_n

            users = random.sample(male_users, male_n) + random.sample(female_users, female_n)

        random.shuffle(users)

        return users

    def load_data(self, user='ID_003', case='A1R', convert=True):
        """ Get x-org, y-org array with ordinates for recording $case from $user

        Return:
        2D np.arrays with equal length x, y as components
        """
        xmin_all_users = []
        ymin_all_users = []
        xmax_all_users = []
        ymax_all_users = []

        filename = self.data_folder + '/{}/{}.txt'.format(user, case)
        if self.use_eye_config == 'RIGHT':
            xy = np.genfromtxt(pjoin(self.data_folder, filename), skip_header=1, usecols=(3, 4))  # right eye
            # print("Right eye",xy[2])
            if self.load_record_duration > 0:
                xy = xy[:self.load_record_duration]

        elif self.use_eye_config == 'BOTH_AVG_GAZE':
            # use for average Gaze position
            xl = np.genfromtxt(pjoin(self.data_folder, filename), skip_header=1, usecols=[1])
            yl = np.genfromtxt(pjoin(self.data_folder, filename), skip_header=1, usecols=[2])

            xr = np.genfromtxt(pjoin(self.data_folder, filename), skip_header=1, usecols=[3])
            yr = np.genfromtxt(pjoin(self.data_folder, filename), skip_header=1, usecols=[4])

            x_lr = [list(xl), list(xr)]
            y_lr = [list(yl), list(yr)]

            x_avg = list(np.nanmean(x_lr, axis=0))  # to find the mean for each two corresponding values in xl, xr lists
            y_avg = list(np.nanmean(y_lr, axis=0))



            # if we want just 8 seconds from the data (we can take 800 gaze points:
            if self.load_record_duration > 0:
                x_avg = x_avg[:self.load_record_duration]
                y_avg = y_avg[:self.load_record_duration]
                # print("the gaze points after avg", len(x_avg))

            xy = np.asarray([x_avg, y_avg]).T

        elif self.use_eye_config == 'LEFT':
            xy = np.genfromtxt(pjoin(self.data_folder, filename), skip_header=1, usecols=(1, 2))  # left eye
            if self.load_record_duration > 0:
                xy = xy[:self.load_record_duration]
        elif self.use_eye_config == 'BOTH_COMBINE_FEAT':
            xyr = np.genfromtxt(pjoin(self.data_folder, filename), skip_header=1, usecols=(3, 4))  # right eye
            xyl = np.genfromtxt(pjoin(self.data_folder, filename), skip_header=1, usecols=(1, 2))  # left eye
            if self.load_record_duration > 0:
                xyr = xyr[:self.load_record_duration]
                xyl = xyl[:self.load_record_duration]
            xy = np.concatenate((np.array(xyl), np.array(xyr)), axis=0)
            # concatenate left and right eyes (same sequence as concatenation of features)
        else:
            raise Exception("Should not happen!")

        if convert:
            # convert angles to pixel
            xy = convert_angles_deg_to_shifted_pixel_coordinates(xy,
                                                                 self.pix_per_mm,
                                                                 self.screen_dist_mm,
                                                                 self.screen_res)



        return xy

    def load_training_trajectories(self):
        # print("train")
        return self.load_trajectories(self.get_cases_training(), self.get_cases())

    def load_training(self, partitions=1):
        """ Loads a list of samples

        :param partitions: int
        Split each trajectory into $partitions samples

        :return: sampleRate: float, dfs: [(np.array(x,y)], id: [int]
        """
        users = self.get_users()
        dfs = []
        ids = []
        for u in users[:self.train_n]:
            df, number_of_nan = self.load_data_no_nan(u, 'A1R')

            # split samples into $splits parts
            dfs.extend(np.array_split(df, partitions))
            c = 'M' if u.endswith(('1', '3')) else 'F'
            ids.extend(np.repeat(c, partitions).tolist())

        return dfs, ids

    def load_testing_trajectories(self):

        return self.load_trajectories(self.get_cases_testing(), self.get_cases())

    def load_testing(self, partitions=1):
        """ Loads a list of samples

        :param partitions: int
        Split each trajectory into $partitions samples

        :return: sampleRate: float, dfs: [(np.array(x,y)], id: [int]
        """
        users = self.get_users()
        dfs = []
        ids = []


        for u in users[self.train_n:self.train_n+self.test_n]:
            df, number_of_nan = self.load_data_no_nan(u, 'A1R')

            # split samples into $splits parts
            dfs.extend(np.array_split(df, partitions))
            c = 'M' if u.endswith(('1', '3')) else 'F'
            ids.extend(np.repeat(c, partitions).tolist())

        return dfs, ids
