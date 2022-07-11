import os.path
import pandas as pd

from schau_mir_in_die_augen.small_functions import load_pickle_file, save_pickle_file, replace_local_path

class DatasetFolder:
    """ This class is for physical Datasets with usersfiles and stimuli.
        It also handles alignment data.
    """

    def __init__(self, data_folder: str, stim_folder: str = None, align_folder: str = None):

        self.data_folder = replace_local_path(data_folder)
        self.stim_folder = replace_local_path(stim_folder)
        self.align_folder = replace_local_path(align_folder)

    def load_alignment(self, stimulus: str) -> pd.DataFrame:
        """ if possible, returns alignment file for selected stimulus.
            This contains x,y-offsets for each user.

        :param stimulus: specific stimulus for eye tracking data
        :return: dictionary with a tuple of offsets for each username.
            "Empty, when there is none."
        """

        if self.align_folder is not None and os.path.isfile(self.align_folder + stimulus + '.csv'):
            user_alignment = pd.read_csv(self.align_folder + stimulus + '.csv')
        else:
            user_alignment = pd.DataFrame({'user': [], 'x': [], 'y': []})

        return user_alignment.set_index('user')

    def save_alignment(self, stimulus: str, offsets: pd.DataFrame):
        """ save alignment for selected stimulus.
            Containing x,y-offsets for each user.

            Will fail, if datasets doesn't support this

        :param stimulus: specific stimulus for eye tracking data
        :param offsets: dictionary with a tuple of offsets for each username.
        """

        if self.align_folder is None:
            raise Exception("Actual Dataset doesn't support alignment saving!")
        else:
            offsets.to_csv(self.align_folder + stimulus + '.csv')

        return
