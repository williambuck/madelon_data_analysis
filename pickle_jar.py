import pandas as pd

samp_3000_bd_df = pd.read_pickle('Data/3000_sample_big_data.p')
samp_6506_bd_df = pd.read_pickle('Data/6506_sample_big_data.p')
results_103_100000rows_df = pd.read_pickle('Data/results_103_100000rows_df.p')

r2_bd_df = pd.read_pickle('Data/mean_r2_bd_sample_1.p')
r2_bd_2_df = pd.read_pickle('Data/mean_r2_bd_sample_2.p')
r2_bd_3_df = pd.read_pickle('Data/mean_r2_bd_sample_3.p')
multi_benchmarks_df = pd.read_pickle('Data/multi_benchmarks_df.p')

top_20_score_benchmark = pd.read_pickle('Data/top_20_score_benchmark.p')
lr_benchmark = pd.read_pickle('Data/lr_benchmark.p')

RFE_cols_score_df = pd.read_pickle('Data/RFE_cols_score_df.p')