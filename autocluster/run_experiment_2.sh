# python metalearning.py \
# --raw_data_path_ls ../data/telco_customer_churn.csv ../data/heart.csv  ../data/loan_data_set.csv ../data/mushrooms.csv ../data/hand_gestures_classification_merged.csv  \
# --json_data_path_ls ../data/processed_data/telco_customer_churn.json ../data/processed_data/heart.json ../data/processed_data/loan_data_set.json ../data/processed_data/mushrooms.json ../data/processed_data/hand_gestures_classification_merged.json \
# --random_seed 27 --n_evaluations 150 --cutoff_time 1000

python metalearning.py \
--raw_data_path_ls ../data/telco_customer_churn.csv ../data/heart.csv  ../data/loan_data_set.csv ../data/hand_gestures_classification_merged.csv  \
--json_data_path_ls ../data/processed_data/telco_customer_churn.json ../data/processed_data/heart.json ../data/processed_data/loan_data_set.json  ../data/processed_data/hand_gestures_classification_merged.json \
--random_seed 27 --n_evaluations 150 --cutoff_time 1000
