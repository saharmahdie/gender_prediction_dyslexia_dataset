#!/bin/bash
# sh eval_gender.sh
# ./eval_gender.sh
## python3 evaluation_heirarchical_dyslexia_gender.py
##python3 evaluation_gender.py  -clf svm --seed 1 --method gender-selection --training_data_ratio 0.8 --user_specific_numbers 19 19 19 19 --use_eye_config BOTH_AVG_FEAT --ivt_threshold 50 --label label_by_dataset --feature_number 8 --feature_list 'mixed_predict_gender'

METHOD="gender-selection"
TRAIN=0.8
EYE='BOTH_AVG_FEAT'
IVT=50

default_block="--method $METHOD --training_data_ratio $TRAIN --use_eye_config $EYE --ivt_threshold $IVT"

# Seeds to do
SEEDstart=1
SEEDend=1000

FEATS=(4)
# feature numbers to do
# FEATS=( 2 12)
# FEATS=( 4 14)
# FEATS=( 6 16)
# FEATS=( 8 18)
# FEATS=( 10 20)
# FEATS=( 22 24)

# User MD FD MH FH
#USUS=( '4 4 14 14'
#       '9 9 9 9'
#       '14 14 4 4'
#       '19 19 19 19')
USUS=( '19 19 0 0')
#USUS=( '69 19 69 19'
#       '19 19 19 19')
#USUS=( '14 14 4 4')
#USUS=( '76 21 69 19')

###'rbfn --rbfn_k 32 ' to predict dyslexia in all 185 users
###'rbfn --rbfn_k 2 ' to predict gender
# classifier with extra arguments
CLFS=('logReg --use_normalization')
#      'rf --use_normalization --max_depth 10 --max_features sqrt --min_samples_leaf 1 --min_samples_split 2'
#      'rbfn --rbfn_k 2 '
#      "svm --use_normalization --svm_gamma $gamma --svm_c $c"
#      'naive_bayes --use_normalization')

for feat in "${FEATS[@]}"; do
  echo
  echo "Starting FEAT = $feat"
  echo
    for usu in "${USUS[@]}"; do
      echo
      echo "Starting USU = $usu"
      echo
        for ((seed=SEEDstart; seed<=SEEDend; seed++)); do
          echo
          echo "Starting SEED = $seed"
          echo
            for clf in "${CLFS[@]}"; do
              echo
              echo "Starting CLF = $clf"
              echo

              # going through all classifier, seeds and feature numbers
              #echo "evaluation_gender.py -clf $clf --seed $seed $default_block --user_specific_numbers $usu --feature_number $feat --feature_list 'dyslexia_prediction_features' --label label_by_dataset"
              #eval "python3 evaluation_gender.py -clf $clf --seed $seed $default_block --user_specific_numbers $usu --feature_number $feat --feature_list 'dyslexia_prediction_features' --label label_by_dataset &"
              echo "evaluation_gender.py -clf $clf --seed $seed $default_block --user_specific_numbers 19 19 0 0 --feature_number $feat --feature_list dyslexic_gender_prediction_features "
              eval "python3 evaluation_gender.py -clf $clf --seed $seed $default_block --user_specific_numbers 19 19 0 0 --feature_number $feat --feature_list dyslexic_gender_prediction_features &"


#              echo "evaluation_gender_Biometric_DS.py -clf $clf --seed $seed $default_block --user_specific_numbers $usu --feature_number $feat"
#              eval "python3 evaluation_gender_Biometric_DS.py -clf $clf --seed $seed $default_block --start_point_load_record_duration 0 --load_record_duration 6000 --feature_number $feat &"

              echo
              echo "Done CLF = $clf"
              echo
            done

            # wait for multiple calls
            wait

          ##  python3 evaluation_gender.py  -clf rf --seed $seed $default_block --ivt_threshold $IVT --train_using_RFE_top_features 5 5
          ## --filter_parameter frame_size 15 pol_order 2
          ##--users_specific_expr True --user_limit 80
          #python3 evaluation_gender.py  -clf rf --seed $seed $default_block --rf_n_estimators 400 --users_specific_expr True --user_limit 80
          #python3 evaluation_gender.py  -clf rf --seed $seed $default_block --rf_n_estimators 400 --label label_by_dataset
          ########python3 evaluation_gender.py  -clf rf --seed $seed $default_block --rf_n_estimators 400
          ########python3 evaluation_gender.py  -clf rf --seed $seed $default_block --rf_n_estimators 400 --label label_by_dataset
          #python3 evaluation_gender.py  -clf rf --seed $seed $default_block --rf_n_estimators 400 --users_specific_expr True --user_limit 52
          #python3 evaluation_gender.py  -clf rf --seed $seed $default_block --rf_n_estimators 400 --label label_by_dataset
          #####
          ####dyslexia
          ##  python3 evaluation_gender.py  -clf rbfn --seed $seed $default_block   --label label_by_dataset --rbfn_k 39
          #  python3 evaluation_gender.py  -clf rbfn --seed $seed $default_block   --rbfn_k 2 --users_specific_expr True --user_limit 80
          ###  python3 evaluation_gender.py  -clf rbfn --seed $seed $default_block   --label label_by_dataset --rbfn_k 39
          #  python3 evaluation_gender.py  -clf rbfn --seed $seed $default_block   --rbfn_k 2 --users_specific_expr True --user_limit 80
          #########  python3 evaluation_gender.py  -clf rbfn --seed $seed $default_block   --label label_by_dataset
          ########  python3 evaluation_gender.py  -clf rbfn --seed $seed $default_block
          ###  python3 evaluation_gender.py  -clf rbfn --seed $seed $default_block   --label label_by_dataset --rbfn_k 39
          #  python3 evaluation_gender.py  -clf rbfn --seed $seed $default_block   --rbfn_k 2 --users_specific_expr True --user_limit 80
          ##  python3 evaluation_gender.py  --clf rbfn --seed $seed $default_block   --label label_by_dataset --rbfn_k 39

          ######
          #  python3 evaluation_gender.py  -clf svm --seed $seed $default_block --ivt_threshold $IVT --label label_by_dataset --svm_gamma $gamma --svm_c $c
          #  python3 evaluation_gender.py  -clf svm --seed $seed $default_block  --svm_gamma $gamma --svm_c $c --users_specific_expr True --user_limit 80
          #  python3 evaluation_gender.py  -clf svm --seed $seed $default_block --ivt_threshold $IVT --label label_by_dataset --svm_gamma $gamma --svm_c $c
          #  python3 evaluation_gender.py  -clf svm --seed $seed $default_block  --svm_gamma $gamma --svm_c $c --users_specific_expr True --user_limit 80
          ########  python3 evaluation_gender.py  -clf svm --seed $seed $default_block --ivt_threshold $IVT --label label_by_dataset --svm_gamma $gamma --svm_c $c
          ########  python3 evaluation_gender.py  -clf svm --seed $seed $default_block --ivt_threshold $IVT --svm_gamma $gamma --svm_c $c
          #  python3 evaluation_gender.py  -clf svm --seed $seed $default_block --ivt_threshold $IVT --label label_by_dataset --svm_gamma $gamma --svm_c $c
          #  python3 evaluation_gender.py  -clf svm --seed $seed $default_block  --svm_gamma $gamma --svm_c $c --users_specific_expr True --user_limit 80
          #  python3 evaluation_gender.py  -clf svm --seed $seed $default_block --ivt_threshold $IVT --label label_by_dataset --svm_gamma $gamma --svm_c $c

          #####
          #  python3 evaluation_gender.py  -clf naive_bayes --seed $seed $default_block --ivt_threshold $IVT --label label_by_dataset
          #  python3 evaluation_gender.py  -clf naive_bayes --seed $seed $default_block  --users_specific_expr True --user_limit 80
          #  python3 evaluation_gender.py  -clf naive_bayes --seed $seed $default_block --ivt_threshold $IVT --label label_by_dataset
          #  python3 evaluation_gender.py  -clf naive_bayes --seed $seed $default_block  --users_specific_expr True --user_limit 80
          #######  python3 evaluation_gender.py  -clf naive_bayes --seed $seed $default_block --ivt_threshold $IVT --label label_by_dataset
          ######  python3 evaluation_gender.py  -clf naive_bayes --seed $seed $default_block --ivt_threshold $IVT
          #  python3 evaluation_gender.py  -clf naive_bayes --seed $seed $default_block --ivt_threshold $IVT --label label_by_dataset
          #  python3 evaluation_gender.py  -clf naive_bayes --seed $seed $default_block  --users_specific_expr True --user_limit 80
          #  python3 evaluation_gender.py  -clf naive_bayes --seed $seed $default_block --ivt_threshold $IVT --label label_by_dataset
          #  python3 evaluation_gender.py  -clf naive_bayes --seed $seed $default_block  --users_specific_expr True --user_limit 80
          ##

          ########## evalute BiometricDS data
          ### max_features auto and first optimized parameters
          ##,oob_score=True
          ###--male_ratio=0.6
          #### 20 seconds data
          #  python3 evaluation_gender_Biometric_DS.py  -clf rf --seed $seed $default_block $extra_attributes  --max_depth 10 --max_features sqrt --min_samples_leaf 4 --min_samples_split 10 --rf_n_estimators 100 --start_point_load_record_duration 0 --load_record_duration 6000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 3 --start_point_load_record_duration 0 --load_record_duration 6000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 2 --start_point_load_record_duration 0 --load_record_duration 6000
          #  python3 evaluation_gender_Biometric_DS.py  -clf svm --seed $seed $default_block $extra_attributes  --svm_gamma 1.3 --svm_c 4.0 --start_point_load_record_duration 0 --load_record_duration 6000
          #  python3 evaluation_gender_Biometric_DS.py  -clf naive_bayes --seed $seed $default_block $extra_attributes  --start_point_load_record_duration 0 --load_record_duration 6000
          #  python3 evaluation_gender_Biometric_DS.py  -clf naive_bayes --seed $seed $default_block $extra_attributes  --start_point_load_record_duration 24000 --load_record_duration 30000
          #
          #  python3 evaluation_gender_Biometric_DS.py  -clf rf --seed $seed $default_block $extra_attributes  --max_depth 10 --max_features sqrt --min_samples_leaf 4 --min_samples_split 10 --rf_n_estimators 100 --start_point_load_record_duration 0 --load_record_duration 12000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 3 --start_point_load_record_duration 0 --load_record_duration 12000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 2 --start_point_load_record_duration 0 --load_record_duration 12000
          #  python3 evaluation_gender_Biometric_DS.py  -clf svm --seed $seed $default_block $extra_attributes  --svm_gamma 1.3 --svm_c 4.0 --start_point_load_record_duration 0 --load_record_duration 12000
          #  python3 evaluation_gender_Biometric_DS.py  -clf naive_bayes --seed $seed $default_block $extra_attributes  --start_point_load_record_duration 0 --load_record_duration 12000
          #
          #  python3 evaluation_gender_Biometric_DS.py  -clf rf --seed $seed $default_block $extra_attributes  --max_depth 10 --max_features sqrt --min_samples_leaf 4 --min_samples_split 10 --rf_n_estimators 100 --start_point_load_record_duration 0 --load_record_duration 18000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 3 --start_point_load_record_duration 0 --load_record_duration 18000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 2 --start_point_load_record_duration 0 --load_record_duration 18000
          #  python3 evaluation_gender_Biometric_DS.py  -clf svm --seed $seed $default_block $extra_attributes  --svm_gamma 1.3 --svm_c 4.0 --start_point_load_record_duration 0 --load_record_duration 18000
          #  python3 evaluation_gender_Biometric_DS.py  -clf naive_bayes --seed $seed $default_block $extra_attributes  --start_point_load_record_duration 0 --load_record_duration 18000

          #  # default
          #  python3 evaluation_gender_Biometric_DS.py  -clf rf --seed $seed $default_block $extra_attributes  --start_point_load_record_duration 0  --load_record_duration 6000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 2 --start_point_load_record_duration 0 --load_record_duration 6000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 3 --start_point_load_record_duration 0 --load_record_duration 6000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 32 --start_point_load_record_duration 0 --load_record_duration 6000
          #  python3 evaluation_gender_Biometric_DS.py  -clf svm --seed $seed $default_block $extra_attributes  --start_point_load_record_duration 0 --load_record_duration 6000
          #
          #  python3 evaluation_gender_Biometric_DS.py  -clf rf --seed $seed $default_block $extra_attributes  --start_point_load_record_duration 0  --load_record_duration 72000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 32  --start_point_load_record_duration 0 --load_record_duration 72000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 2  --start_point_load_record_duration 0 --load_record_duration 72000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 7  --start_point_load_record_duration 0 --load_record_duration 72000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 13  --start_point_load_record_duration 0 --load_record_duration 72000
          #  python3 evaluation_gender_Biometric_DS.py  -clf svm --seed $seed $default_block $extra_attributes  --start_point_load_record_duration 0 --load_record_duration 72000
          #### 4 min data
          ##  ## second optimized parameters
          #  python3 evaluation_gender_Biometric_DS.py  -clf rf --seed $seed $default_block $extra_attributes  --max_depth 10 --max_features sqrt --min_samples_leaf 4 --min_samples_split 2 --rf_n_estimators 100 --start_point_load_record_duration 0 --load_record_duration 72000
          #  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 7  --start_point_load_record_duration 0 --load_record_duration 72000
          ##  python3 evaluation_gender_Biometric_DS.py  -clf rbfn --seed $seed $default_block  --rbfn_k 13  --start_point_load_record_duration 0 --load_record_duration 72000
          #  python3 evaluation_gender_Biometric_DS.py  -clf svm --seed $seed $default_block $extra_attributes  --svm_gamma 0.2 --svm_c 0.4 --start_point_load_record_duration 0 --load_record_duration 72000
          #  python3 evaluation_gender_Biometric_DS.py  -clf naive_bayes --seed $seed $default_block $extra_attributes  --start_point_load_record_duration 0 --load_record_duration 72000
          #
          #  python3 evaluation_gender_Biometric_DS.py  -clf naive_bayes --seed $seed $default_block $extra_attributes  --start_point_load_record_duration 72000 --load_record_duration 144000



          # python3 evaluation_gender_Biometric_DS.py --method gender-selection -clf svm --seed 10 $default_block $extra_attributes  --svm_gamma 1.3 --svm_c 4.0 --start_point_load_record_duration 0 --load_record_duration 6000

          echo
          echo "Done SEED = $seed"
          echo
        done
      echo
      echo "Done USU = $usu"
      echo
    done
  echo
  echo "Done FEAT = $feat"
  echo
done

echo '>>> SCRIPT COMPLETLY DONE! <<<'
echo
echo "Started with:"
# shellcheck disable=SC2128
echo " Feat $FEATS"
# shellcheck disable=SC2128
echo " USU $USUS"
echo " SEED $SEEDstart"
# shellcheck disable=SC2128
echo " CLF $CLFS"

# play sound (not shure it will work, but not important if it does so
paplay '/usr/share/sounds/ubuntu/stereo/message.ogg'
