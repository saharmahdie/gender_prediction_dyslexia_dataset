## sh eval_hierarchical_classification_gender.sh
##  python3 evaluation_heirarchical_dyslexia_gender.py -clf rbfn --seed 1 --method gender-selection --training_data_ratio 0.8 --user_specific_numbers 69 19 69 19 --use_eye_config BOTH_AVG_FEAT --ivt_threshold 50
## python3 Ensemble_Classifier_Voting_or_Weighted.py --method gender-selection --ivt_threshold 50 --seed 1 --training_data_ratio 0.8 --user_specific_numbers 7 7 7 7 --use_eye_config BOTH_AVG_FEAT

CLF=$1
METHOD=${2-"gender-selection"}


ITERS=1000
for i in $(seq 1 $ITERS); do
#  python3 evaluation_heirarchical_dyslexia_gender.py --method gender-selection --ivt_threshold 50 --classifier rbfn --seed $i --training_data_ratio 0.8 --user_specific_numbers 69 19 69 19 --use_eye_config BOTH_AVG_FEAT --H_D_train_user_num  3 3 11 11
#  python3 evaluation_heirarchical_dyslexia_gender.py --method gender-selection --ivt_threshold 50 --classifier rbfn --seed $i --training_data_ratio 0.8 --user_specific_numbers 69 19 69 19 --use_eye_config BOTH_AVG_FEAT --H_D_train_user_num  11 11 3 3
  python3 evaluation_heirarchical_dyslexia_gender.py --method gender-selection --ivt_threshold 50 --classifier rbfn --seed $i --training_data_ratio 0.8 --user_specific_numbers 69 19 69 19 --use_eye_config BOTH_AVG_FEAT --H_D_train_user_num  7 7 7 7
#  python3 evaluation_heirarchical_dyslexia_gender.py --method gender-selection --ivt_threshold 50 --classifier rbfn --seed $i --training_data_ratio 0.8 --user_specific_numbers 19 19 19 19 --use_eye_config BOTH_AVG_FEAT --H_D_train_user_num 15 15 15 15
#  python3 evaluation_heirarchical_dyslexia_gender.py --method gender-selection --ivt_threshold 50 --classifier rbfn --seed $i --training_data_ratio 0.8 --user_specific_numbers 18 18 18 18 --use_eye_config BOTH_AVG_FEAT --H_D_train_user_num 15 15 15 15
#  python3 evaluation_heirarchical_dyslexia_gender.py --method gender-selection --ivt_threshold 50 --classifier rbfn --seed $i --training_data_ratio 0.8 --user_specific_numbers 9 9 9 9 --use_eye_config BOTH_AVG_FEAT --H_D_train_user_num 15 15 15 15
#  python3 evaluation_heirarchical_dyslexia_gender.py --method gender-selection --ivt_threshold 50 --classifier rbfn --seed $i --training_data_ratio 0.8 --user_specific_numbers 14 14 4 4 --use_eye_config BOTH_AVG_FEAT --H_D_train_user_num 15 15 15 15
#  python3 evaluation_heirarchical_dyslexia_gender.py --method gender-selection --ivt_threshold 50 --classifier rbfn --seed $i --training_data_ratio 0.8 --user_specific_numbers 4 4 14 14 --use_eye_config BOTH_AVG_FEAT --H_D_train_user_num 15 15 15 15

done
