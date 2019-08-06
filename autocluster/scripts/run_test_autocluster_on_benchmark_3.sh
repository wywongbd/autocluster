cd ..

# warmstart only
# python test_autocluster_on_benchmark.py \
# --benchmark_metafeatures_table_path metaknowledge/benchmark_silhouette_metafeatures_table.csv \
# --log_dir_prefix benchmark_experiment_1 \
# --optimizer smac \
# --warmstart 1 \
# --test_size 0.1666 \
# --n_folds 3 \
# --random_seed 27 \
# --n_evaluations 25 \
# --cutoff_time 100 \
# --warmstart_n_neighbors 5 \
# --warmstart_top_n 5

# bayesian optimization only
# python test_autocluster_on_benchmark.py \
# --benchmark_metafeatures_table_path metaknowledge/benchmark_silhouette_metafeatures_table.csv \
# --log_dir_prefix benchmark_experiment_2 \
# --optimizer smac \
# --warmstart 0 \
# --test_size 0.1666 \
# --n_folds 3 \
# --random_seed 27 \
# --n_evaluations 100 \
# --cutoff_time 100 \
# --warmstart_n_neighbors 1 \
# --warmstart_top_n 1 

# warmstart + bayesian optimization
python test_autocluster_on_benchmark.py \
--benchmark_metafeatures_table_path metaknowledge/benchmark_silhouette_metafeatures_table_v0.csv \
--log_dir_prefix benchmark_experiment_3_v0 \
--optimizer smac \
--warmstart 1 \
--test_size 0.1666 \
--n_folds 3 \
--random_seed 27 \
--n_evaluations 100 \
--cutoff_time 100 \
--warmstart_n_neighbors 5 \
--warmstart_top_n 5 