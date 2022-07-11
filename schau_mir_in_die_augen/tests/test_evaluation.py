""" Demonstration of Train and Test """

from unittest import TestCase

from schau_mir_in_die_augen.evaluation.script_helper import call_train, call_test


class TestEvaluation(TestCase):

    #: datasets to test
    datasets = []
    #: classifiers to test
    classifiers = []
    #: methods to test
    methods = []
    #: aims for prediction to test
    labels = []

    #: name of file to save trained algorithm
    model_file_name = '[model]test_evaluation_model.pickle'
    #: number of users to use
    user_limit = 10

    def setUp(self):
        """ Setup the Test Scenarios.

        All different combinations will be tested.
        """

        self.datasets = ['demo-data', 'demo-user', 'demo-gender', 'demo-user-gender']
        self.classifiers = ['logReg']
        self.methods = ['score-level']
        self.labels = ['user', 'gender']

        self.model_file_name = '[model]test_evaluation_model.pickle'
        self.user_limit = 10

    def test_evaluation(self):
        """ Will go trough all selected evaluations and run train and test """

        results = dict()

        for label in self.labels:
            for dataset in self.datasets:
                for classifier in self.classifiers:
                    for method in self.methods:

                        # training
                        train_results = call_train(dataset=dataset,
                                                   classifier=classifier,
                                                   method=method,
                                                   label=label,
                                                   user_limit=self.user_limit,
                                                   modelfile=self.model_file_name,
                                                   test_train=True)
                        # evaluation
                        test_results = call_test(dataset=dataset,
                                                 classifier=classifier,
                                                 method=method,
                                                 label=label,
                                                 user_limit=self.user_limit,
                                                 modelfile=self.model_file_name)

                        # save important data
                        results.update({'{ds} {clf} {met} {lab}'.format(
                            ds=dataset, clf=classifier, met=method, lab=label): {'test': test_results,
                                                                                 'train': train_results}})

        for result in results:
            print(result, results[result])
