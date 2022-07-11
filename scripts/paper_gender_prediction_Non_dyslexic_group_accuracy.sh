## keep the results folder filename in evaluation_gender.py = "Gender_prediction_paper_accuracy"
ITERS=1000
for i in $(seq 1 $ITERS); do
  echo
  echo "Run number = $i"
  echo
  python3 evaluation_gender.py  -clf svm --seed $i --method gender-selection --training_data_ratio 0.8 --user_specific_numbers 0 0 19 19 --use_eye_config BOTH_AVG_FEAT --ivt_threshold 50 --feature_number 4 --feature_list 'healthy_gender_prediction_features' --use_normalization --svm_gamma 0.1 --svm_c 0.1
  python3 evaluation_gender.py  -clf rf --seed $i --method gender-selection --training_data_ratio 0.8 --user_specific_numbers 0 0 19 19 --use_eye_config BOTH_AVG_FEAT --ivt_threshold 50 --feature_number 4 --feature_list 'healthy_gender_prediction_features' --use_normalization --max_depth 10 --max_features sqrt --min_samples_leaf 1 --min_samples_split 2
  python3 evaluation_gender.py  -clf rbfn --seed $i --method gender-selection --training_data_ratio 0.8 --user_specific_numbers 0 0 19 19 --use_eye_config BOTH_AVG_FEAT --ivt_threshold 50 --feature_number 4 --feature_list 'healthy_gender_prediction_features' --rbfn_k 2
  python3 evaluation_gender.py  -clf logReg --seed $i --method gender-selection --training_data_ratio 0.8 --user_specific_numbers 0 0 19 19 --use_eye_config BOTH_AVG_FEAT --ivt_threshold 50 --feature_number 4 --feature_list 'healthy_gender_prediction_features' --use_normalization
  python3 evaluation_gender.py  -clf naive_bayes --seed $i --method gender-selection --training_data_ratio 0.8 --user_specific_numbers 0 0 19 19 --use_eye_config BOTH_AVG_FEAT --ivt_threshold 50 --feature_number 4 --feature_list 'healthy_gender_prediction_features' --use_normalization
done