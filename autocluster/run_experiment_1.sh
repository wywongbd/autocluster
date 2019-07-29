# python metalearning.py \
# --raw_data_path_ls ../data/fertility.csv ../data/breast_cancer.csv ../data/bank-additional-full.csv ../data/glass.csv ../data/nasa.csv ../data/iris.csv ../data/banana.csv  \
# --json_data_path_ls ../data/processed_data/fertility.json ../data/processed_data/breast_cancer.json ../data/processed_data/bank-additional-full.json  ../data/processed_data/glass.json ../data/processed_data/nasa.json ../data/processed_data/iris.json ../data/processed_data/banana.json  \
# --random_seed 27 --n_evaluations 150 --cutoff_time 1000

python metalearning.py \
--raw_data_path_ls ../data/fertility.csv ../data/bank-additional-full.csv ../data/nasa.csv ../data/banana.csv  \
--json_data_path_ls ../data/processed_data/fertility.json ../data/processed_data/bank-additional-full.json ../data/processed_data/nasa.json ../data/processed_data/banana.json  \
--random_seed 27 --n_evaluations 150 --cutoff_time 1000
