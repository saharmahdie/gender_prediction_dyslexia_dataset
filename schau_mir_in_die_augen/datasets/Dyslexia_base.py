import os
import logging
import numpy as np
from os.path import join as pjoin

from schau_mir_in_die_augen.datasets.DatasetBase import DatasetBase
from schau_mir_in_die_augen.datasets.DatasetFolder import DatasetFolder

class DyslexiaBase(DatasetBase, DatasetFolder):

    # todo: Dyslexia is very confusing. I think all could be combined in one class.
    # todo: The class does a lot in initializing.

    def __init__(self, user_limit=None, seed=42, training_data=0.8, male_ratio: float = None, user_specific_numbers=False):
        """

        :param user_limit: int
        to choose the number of users to load
        :param training_data: float
        can be used to define the ratio of data to be kept in training (remaining will go to testing)

        """
        DatasetBase.__init__(self,
                             user_limit=user_limit, seed=seed,
                             sample_rate=100,
                             data_kind='angle_deg',
                             dataset_name='Dyslexia')

        DatasetFolder.__init__(self, data_folder='[smida]data/Dyslexia/RecordingData')

        self.dataset_key = 'dyslexia'

        self.train_ratio = training_data
        self.user_specific_numbers = user_specific_numbers

        # Screen dimensions (w × h): 474 × 297 mm
        self.screen_size_mm = np.asarray([474, 297])


        # Screen resolution (w × h): 1680 × 1050 pixels
        self.screen_res = np.asarray([1680, 1050])

        self.pix_per_mm = self.screen_res / self.screen_size_mm
        # Subject's distance from screen: 450 mm
        self.screen_dist_mm = 450  # (given)

        # extra scaling factor
        self.scaling_factor = 0.2

        self.male_ratio = male_ratio
        if male_ratio is None and os.path.exists(self.data_folder):  # Folder could be empty todo: stop earlier
            genders = [self.get_gender(user) for user in self.get_users()]
            self.male_ratio = genders.count('M') / len(genders)

        if self.user_limit:
            # if use_limit is defined then will divide these samples into training and testing data
            # based on training_ratio value
            self.test_n = int(np.round(self.user_limit * (1 - self.train_ratio)))
            self.train_n = int(self.user_limit - self.test_n)
        elif os.path.exists(self.data_folder):
            # else all data will be used and divided into training and test data using the train_ratio value
            users = len([u for u in os.listdir(self.data_folder) if os.path.isdir(pjoin(self.data_folder, u))])
            self.test_n = int(np.round(users * (1 - self.train_ratio)))
            self.train_n = int(users - self.test_n)
        else:
            print('Dyslexia could not be initialized. Data is missing.')  # todo: this should be Done earlier

    def get_screen_params(self):
        return {'pix_per_mm': self.pix_per_mm,
                'screen_dist_mm': self.screen_dist_mm,
                'screen_res': self.screen_res}

    def get_cases(self):
        """ List of recordings per user

        :return: list of recordings per user
        """
        # There is only one case
        return ['A1R']

    def get_stimulus(self, case='A1R'):
        """ Relative filename of stimulus

        :param case: case from get_cases
        :return: string
           Relative path to stimulus
        """
        return [self.screen_res[0], self.screen_res[1]]

    def get_gender(self, user: str):
        return 'M' if user.endswith(('1', '3')) else 'F'

    def get_label(self, user: str):
        """ can be modified to return any classification """

        return 'healthy' if user.endswith(('3', '4')) else 'dyslexic'


    def get_cases_training(self):

        logger = logging.getLogger(__name__)
        users = self.get_users()

        num_dm = round(self.train_ratio*len([u for u in users if u.endswith('1')]))
        num_df = round(self.train_ratio*len([u for u in users if u.endswith('2')]))
        num_hm = round(self.train_ratio*len([u for u in users if u.endswith('3')]))
        num_hf = round(self.train_ratio*len([u for u in users if u.endswith('4')]))

        dm = [u for u in users if u.endswith('1')]
        df = [u for u in users if u.endswith('2')]
        hm = [u for u in users if u.endswith('3')]
        hf = [u for u in users if u.endswith('4')]

        x = dm[:num_dm] + df[:num_df] + hm[:num_hm] + hf[:num_hf]  ## to predict gender mixed or not mixed group

        return x      ## to predict gender mixed or not mixed group


    def get_cases_testing(self):
        # if we use male_ratio argument, we have to balance the healthy and dyslexic ratio like we did in get_cases_train()
        logger = logging.getLogger(__name__)
        users = self.get_users()
        # genders = [self.get_gender(user) for user in users]
        train_users = self.get_cases_training()



        num_dm = round((1-self.train_ratio)*len([u for u in users if u.endswith('1')]))
        num_df = round((1-self.train_ratio)*len([u for u in users if u.endswith('2')]))
        num_hm = round((1-self.train_ratio)*len([u for u in users if u.endswith('3')]))
        num_hf = round((1-self.train_ratio)*len([u for u in users if u.endswith('4')]))
        # print("num_dm",num_dm, "num_df",num_df, "num_hm",num_hm, "num_hf",num_hf)

        # ensure that there is no train user in the test user
        dm = [u for u in users if (u.endswith('1') and u not in train_users)]
        df = [u for u in users if (u.endswith('2') and u not in train_users)]
        hm = [u for u in users if (u.endswith('3') and u not in train_users)]
        hf = [u for u in users if (u.endswith('4') and u not in train_users)]

        x = dm[:num_dm] + df[:num_df] + hm[:num_hm] + hf[:num_hf]  ## to predict gender mixed or not mixed group

        return x

    def modify_trajectory(self, trajectory):
        """ scale trajectory when loaded """

        trajectory.scale(self.scaling_factor)

        return trajectory
