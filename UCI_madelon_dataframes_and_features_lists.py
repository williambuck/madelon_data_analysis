import pandas as pd

# Assigning file paths to variables. 
train_data_uci_madelon = './web_madelon_data/madelon_train.data.txt'
train_label_uci_madelon = './web_madelon_data/madelon_train.labels.txt'
val_data_uci_madelon = './web_madelon_data/madelon_valid.data.txt'
val_label_uci_madelon = './web_madelon_data/madelon_valid.labels.txt'
test_data_uci_madelon = './web_madelon_data/madelon_test.data.txt'
params_uci_madelon = './web_madelon_data/madelon.param.txt'


# Creating dataframes for the train, test, and val datasets.
train_uci_df = pd.read_csv(train_data_uci_madelon, delimiter=' ', header=None).drop(500, axis=1)
test_uci_df = pd.read_csv(test_data_uci_madelon, delimiter=' ', header=None).drop(500, axis=1)
val_uci_df = pd.read_csv(val_data_uci_madelon, delimiter=' ', header=None).drop(500, axis=1)


# Creating column names for all of the uci dataframes.
feature_col_names = ['feat_{}'.format(i) for i in range(0,500)]
train_uci_df.columns = feature_col_names
test_uci_df.columns = feature_col_names
val_uci_df.columns = feature_col_names


y_train = pd.read_csv(train_label_uci_madelon, header=None)
y_val = pd.read_csv(val_label_uci_madelon, header=None)

y_train.columns = ['target']
y_val.columns = ['target']


# Final DataFrames with labels
train_uci_df = pd.merge(train_uci_df, y_train, left_index=True, right_index=True)
val_uci_df = pd.merge(val_uci_df, y_val, left_index=True, right_index=True)

# 20 Columns (5 important, 15 redundant)
top_20_real_features = [28,48,64,105,128,153,241,281,318,336,338,378,433,442,451,453,455,472,475,493]
top_14_elasticnet_real_features = [64,105,153,241,318,336,338,378,442,453,455,472,475,493]
top_12_real_features = [48,64,105,128,241,318,336,338,378,442,453,475]
top_7_real_features = [64, 128, 241, 336, 338, 378, 475]
top_5_real_features = [64, 336, 338, 378, 475]

fourth_best_score_list = [28, 48, 105, 128, 153, 281, 318, 336, 338, 378, 442, 451, 455, 472, 475]
third_best_score_list = [28,48,64,105,128,153,241,318,336,338,378,433,442,451,453,455,475,493]
second_best_score_list = [378, 153, 442, 336, 451, 48, 28, 318, 475, 455]
best_score_list = [336, 475, 442, 153, 318, 48, 451, 455]
top_5_best_score = [153, 451, 48, 475, 455]

top_scores_list_of_lists = [top_5_best_score, best_score_list, second_best_score_list, third_best_score_list]