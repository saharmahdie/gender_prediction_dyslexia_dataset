from schau_mir_in_die_augen.evaluation.base_selection import parse_arguments

from scripts.train import main as train_main
from scripts.evaluation import main as test_main


def prepare_input(dataset: str, classifier: str, method: str, modelfile: str,
                  label: str = None, user_limit: int = None, test_train: bool = False, seed: int = None,
                  ivt_threshold: int = None, ivt_min_fix_time: int = None,
                  filter_data: str = None, filter_type: str = None, filter_parameter: list = None) -> list:
    """ return a list with arguments for evaluation and training"""

    input_args = ['--dataset', dataset,
                  '--classifier', classifier,
                  '--method', method,
                  '--modelfile', modelfile,
                  ]

    if label is not None:
        input_args.extend(['--label', label])
    if user_limit is not None:
        input_args.extend(['--user_limit', str(user_limit)])
    if test_train:
        input_args.extend(['--test_train'])
    if seed is not None:
        input_args.extend(['--seed', str(seed)])

    if ivt_threshold is not None:
        input_args.extend(['--ivt_threshold', str(ivt_threshold)])
    if ivt_min_fix_time is not None:
        input_args.extend(['--ivt_min_fix_time', str(ivt_min_fix_time)])

    if filter_data is not None:
        input_args.extend(['--filter_data', filter_data])
    if filter_type is not None:
        input_args.extend(['--filter_type', filter_type])
    if filter_parameter is not None:
        input_args.extend(['--filter_parameter', *filter_parameter])

    return input_args

def print_info(info_tag: str, dataset: str, classifier: str, method: str, modelfile: str,
               label: str = None):

    name_to_match_other = info_tag

    print('#####' + '#' * len(info_tag) + '##### dataset: ' + dataset)
    print('#####' + '#' * len(info_tag) + '##### clf    : ' + classifier)
    print('#### ' + name_to_match_other + ' #### method : ' + method)
    print('#####' + '#' * len(info_tag) + '##### label  : ' + label)
    print('#####' + '#' * len(info_tag) + '##### mf     : ' + modelfile)


def call_train(dataset: str, classifier: str, method: str, modelfile: str,
               label: str = None, user_limit: int = None, test_train: bool = False, seed: int = None,
               ivt_threshold: int = None, ivt_min_fix_time: int = None,
               filter_data: str = None, filter_type: str = None, filter_parameter: list = None):

    input_args = prepare_input(dataset=dataset, classifier=classifier, method=method, modelfile=modelfile,
                               label=label, user_limit=user_limit, test_train=test_train, seed=seed,
                               ivt_threshold=ivt_threshold, ivt_min_fix_time=ivt_min_fix_time,
                               filter_data=filter_data, filter_type=filter_type, filter_parameter=filter_parameter)

    # make possible to look for specific parameters
    #   see get_conditional_parser
    args = parse_arguments(input_args + ['--more_parameters', 'train'])

    print_info(info_tag='Training',
               dataset=dataset, classifier=classifier, method=method, modelfile=modelfile, label=label)
    print(args)

    return train_main(args)


def call_test(dataset: str, classifier: str, method: str, modelfile: str,
              label: str = None, user_limit: int = None, seed: int = None,
              ivt_threshold: int = None, ivt_min_fix_time: int = None,
              filter_data: str = None, filter_type: str = None, filter_parameter: list = None):

    input_args = prepare_input(dataset=dataset, classifier=classifier, method=method, modelfile=modelfile,
                               label=label, user_limit=user_limit, seed=seed,
                               ivt_threshold = ivt_threshold, ivt_min_fix_time = ivt_min_fix_time,
                               filter_data=filter_data, filter_type=filter_type, filter_parameter=filter_parameter)
    args = parse_arguments(input_args)

    print_info(info_tag='Testing',
               dataset=dataset, classifier=classifier, method=method, modelfile=modelfile, label=label)
    print(args)

    return test_main(args)
