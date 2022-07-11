## This script used for printing the mean accuracy and the STD over all the seeds (runs)
import sys
sys.path.append('../')
import json
import glob
import numpy as np

accuracy_rf = []
seeds_rf = []
accuracy_rbfn = []
seeds_rbfn = []
accuracy_svm = []
seeds_svm = []
accuracy_naive = []
seeds_naive = []
accuracy_logReg = []
seeds_logReg = []

files_logReg =glob.glob("./results/Gender_prediction_paper_accuracy/predict_MF_with_healthy_gender_prediction_features_trainRatio_0.8/featuers_score_MD0+FD0+MH19+FH19/BOTH_AVG_FEAT/gender-selection/logReg/IVT_50/FEATS_4/*.json")
files_rf = glob.glob("./results/Gender_prediction_paper_accuracy/predict_MF_with_healthy_gender_prediction_features_trainRatio_0.8/featuers_score_MD0+FD0+MH19+FH19/BOTH_AVG_FEAT/gender-selection/rf/IVT_50/FEATS_4/*.json")
files_rbfn = glob.glob("./results/Gender_prediction_paper_accuracy/predict_MF_with_healthy_gender_prediction_features_trainRatio_0.8/featuers_score_MD0+FD0+MH19+FH19/BOTH_AVG_FEAT/gender-selection/rbfn/IVT_50/FEATS_4/*.json")
files_svm = glob.glob("./results/Gender_prediction_paper_accuracy/predict_MF_with_healthy_gender_prediction_features_trainRatio_0.8/featuers_score_MD0+FD0+MH19+FH19/BOTH_AVG_FEAT/gender-selection/svm/IVT_50/FEATS_4/*.json")
files_naive = glob.glob("./results/Gender_prediction_paper_accuracy/predict_MF_with_healthy_gender_prediction_features_trainRatio_0.8/featuers_score_MD0+FD0+MH19+FH19/BOTH_AVG_FEAT/gender-selection/naive_bayes/IVT_50/FEATS_4/*.json")

for file in files_logReg:
    with open(file, 'r') as f:
        data = json.load(f)
    ##accuracy_rf.append(data['Accuracy_unmerged_balanced'])
    accuracy_logReg.append(data['Accuracy_balanced'])
    seeds_logReg.append(data['config']['seed'])
print("*************************************************************")
print("Balanced Accuracy of LogReg is ::",np.mean(accuracy_logReg))
print("STD of LogReg accuracy is ::",np.std(accuracy_logReg))
print("Runs Number of LogReg is ::",len(accuracy_logReg))
print("*************************************************************")

for file in files_rf:
    with open(file, 'r') as f:
        data = json.load(f)
    accuracy_rf.append(data['Accuracy_balanced'])
    seeds_rf.append(data['config']['seed'])
print("Balanced Accuracy of Random Forest is ::",np.mean(accuracy_rf))
print("STD of Random Forest accuracy is ::",np.std(accuracy_rf))
print("Runs Number of Random Forest is ::",len(accuracy_rf))
print("*************************************************************")

for file in files_rbfn:
    with open(file, 'r') as f:
        data = json.load(f)
    accuracy_rbfn.append(data['Accuracy_balanced'])
    seeds_rbfn.append(data['config']['seed'])
print("Balanced Accuracy of RBFN is ::",np.mean(accuracy_rbfn))
print("STD of RBFN accuracy is ::",np.std(accuracy_rbfn))
print("Runs Number of RBFN is ::",len(accuracy_rbfn))
print("*************************************************************")

for file in files_svm:
    with open(file, 'r') as f:
        data = json.load(f)
    accuracy_svm.append(data['Accuracy_balanced'])
    seeds_svm.append(data['config']['seed'])
print("Balanced Accuracy of SVM is ::",np.mean(accuracy_svm))
print("STD of SVM accuracy is ::",np.std(accuracy_svm))
print("Runs Number of SVM is ::",len(accuracy_svm))
print("*************************************************************")

for file in files_naive:
    with open(file, 'r') as f:
        data = json.load(f)
    accuracy_naive.append(data['Accuracy_balanced'])
    seeds_naive.append(data['config']['seed'])
print("Balanced Accuracy of NB is ::",np.mean(accuracy_naive))
print("STD of NB accuracy is ::",np.std(accuracy_naive))
print("Runs Number of NB is ::",len(accuracy_naive))
print("*************************************************************")

