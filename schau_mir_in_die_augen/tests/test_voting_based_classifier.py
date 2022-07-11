from unittest import TestCase
# import numpy as np

from scripts.evaluation_heirarchical_dyslexia_gender import voting_based_classifier, weighting_based_classifier

class TestDatasets(TestCase):
## this test incase deactivate CLF1 means take the true labels
    def setUp(self):

        self.res = {
            'True_Lable': ['healthy', 'dyslexic', 'healthy', 'dyslexic', 'healthy', 'dyslexic', 'healthy', 'dyslexic'],
            'Predicted_Lable': ['healthy', 'healthy', 'healthy', 'healthy', 'dyslexic', 'dyslexic', 'dyslexic',
                                'dyslexic']}
        self.res_healthy = {'True_Lable': ['F', 'M', 'F', 'M', 'F', 'M', 'M', 'M'],
                       'Predicted_Lable': ['M', 'M', 'F', 'M', 'F', 'M', 'M', 'M']}
        self.res_dyslexic = {'True_Lable': ['F', 'M', 'F', 'M', 'F', 'M', 'M', 'M'],
                        'Predicted_Lable': ['M', 'M', 'F', 'M', 'F', 'M', 'M', 'M']}


    def test_Herarchical_accuracy(self):

        voting_accuracy=voting_based_classifier(res=self.res, res_healthy=self.res_healthy, res_dyslexic=self.res_dyslexic)
        true_res = [0.875, 0.8333333333333333, 0.75, 0.75, 1.0, 1.0]
        for i in range(len(true_res)):
            self.assertEqual(true_res[i], voting_accuracy[i])
            #print("i", i, "true_res[i]", true_res[i], "voting_accuracy[i]", voting_accuracy[i], "assertEqual", self.assertEqual(true_res[i], voting_accuracy[i]) )
