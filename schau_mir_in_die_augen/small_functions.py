import os
import pickle

import pandas as pd

from pathlib import Path


def load_pickle_file(file_name: str):
    """ load content from a pickle file

    :param file_name: name of file
        If you use a [keyword] at the beginning it will be replaced. See replace_local_path.
    """

    file_name = replace_local_path(file_name)

    try:
        with open(file_name, 'rb') as f:
            content = pickle.load(f)
    except Exception:
        print(f'Failed loading file: "{file_name}" in "{os.getcwd()}"')
        raise
    return content


def save_pickle_file(file_name: str, content):
    """ save content in a pickle file

    :param file_name: name of file
        If you use a [keyword] at the beginning it will be replaced. See replace_local_path.
    :param content: content to be saved
    """

    file_name = replace_local_path(file_name)

    try:
        with open(file_name, 'wb') as f:
            pickle.dump(content, f)
    except FileNotFoundError:
        print(f'There is no directory "{file_name}"')
    except Exception:
        print(f'Failed to save file: "{file_name}" from "{os.getcwd()}"')
        raise


def replace_local_path(file_name: str) -> str:
    """ Will search for keyword at the beginning and replace it with actual filepath.

    :param file_name:
        '[model]' will be replaced by path to model folder (with ending slash)
        '[bokeh_settings]' will be replaced by user folder
    :return file_name: with changed keyword, or like before
    """

    # order has to be rising length

    bokeh_main = '[bet]'
    bokeh_user = '[bus]'
    smida = '[smida]'
    model = '[model]'

    if file_name is not None and len(file_name) >= len(bokeh_main):
        if file_name[:len(bokeh_main)] == bokeh_main:
            file_name = get_top_level_path() + '/settings/bokeh/' + file_name[len(bokeh_main):]
        elif file_name[:len(bokeh_user)] == bokeh_user:
            file_name = str(Path.home()) + '/SMIDAsettings/bokeh/' + file_name[len(bokeh_user):]

        elif len(file_name) >= len(model):
            if file_name[:len(model)] == model:
                file_name = get_top_level_path() + '/model/' + file_name[len(model):]
            elif file_name[:len(smida)] == smida:
                file_name = get_top_level_path() + '/' + file_name[len(smida):]

    return file_name


def get_top_level_path() -> str:
    """
    :return path to top level folder of SMIDA:
        (without slash at the end)
    """
    # get path of this file (remove symbolic links)
    file_path = os.path.realpath(__file__)
    # step 3 folders up. This has to be changed, when this file is moved.
    file_path = os.path.dirname(os.path.dirname(file_path))

    return file_path


def get_files(foldername: str) -> list:
    """ Scan some folder and return list of local filenames
    :param foldername: string
        It can start with the following keywords:
        "[model]" for model folder
        "[bokeh_settings]" for user settings
    """

    assert foldername in ['[model]', '[bet]', '[bus]']
    folderpath = replace_local_path(foldername)

    if not os.path.isdir(folderpath):
        return []

    filenames = os.listdir(folderpath)

    return [foldername + filename for filename in filenames if filename[0] != '.']


def select_probabilities(labeled_predicted_probabilities: dict, user: str, classifier: str):
    """ Select probabilities from different users and specific classifier

    :param labeled_predicted_probabilities: dict with DataFrames of prediction values for each user
    :param user: Name of selected user
    :param classifier: Name of classifier
        'all mean': mean of all probabilities for this user
        'clf mean': mean of the classifier results (differs when classifiers have different numbers of subsets)
        'fix': only fixation (hardcode for now)
        'sac': only saccades (hardcode for now)

    """

    labeled_predicted_probabilities_user = labeled_predicted_probabilities[user]

    # todo: fix this hard code - atm only for saccades/fixations - names should be the same
    clf_hard_names = ['saccade', 'fixation']

    if classifier == 'all mean':
        selected_probabilities = labeled_predicted_probabilities_user.drop(columns=['sample_type'])
    elif classifier == 'clf mean':
        # calculate mean for every classifer
        selected_probabilities = pd.concat([
            labeled_predicted_probabilities_user[labeled_predicted_probabilities_user['sample_type'] == clf_name].mean(
                axis=0)
            for clf_name in clf_hard_names], axis=1).T
        selected_probabilities['sample_type'] = clf_hard_names  # this gets dropped by .mean
        selected_probabilities.set_index(keys='sample_type', inplace=True)
    elif classifier == 'sac':
        selected_probabilities = labeled_predicted_probabilities_user[
            labeled_predicted_probabilities_user['sample_type'] == 'saccade'].drop(columns=['sample_type'])
    elif classifier == 'fix':
        selected_probabilities = labeled_predicted_probabilities_user[
            labeled_predicted_probabilities_user['sample_type'] == 'fixation'].drop(columns=['sample_type'])
    else:
        raise Exception('"{}" is not a valid choice!'.format(classifier))

    return selected_probabilities
