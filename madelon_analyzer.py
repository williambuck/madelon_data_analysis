import csv
import psycopg2 as pg2
from psycopg2.extras import RealDictCursor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

from sklearn.model_selection import GridSearchCV, ShuffleSplit, StratifiedShuffleSplit, \
                                    train_test_split, StratifiedKFold, KFold
    
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, \
                                 LogisticRegressionCV, LassoCV, \
                                 ElasticNetCV, RidgeCV, Lasso, ElasticNet
        

from sklearn.feature_selection import SelectKBest, \
                                      SelectFromModel, \
                                      RFE, SelectPercentile, \
                                      f_regression
            
from sklearn.ensemble import GradientBoostingClassifier

from scipy.stats.mstats import normaltest
            
from sklearn.pipeline import Pipeline
from IPython.display import display
from itertools import combinations


class madelon_analyzer:
    '''This class is intended to make analyzing the Madelon data easier. Instantiation
    requires a training dataframe and a validation dataframe.'''
    
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    
    features = []
    results = {}
    params = {}
    model = ()
    best_params_ = {}

    def __init__(self, train_df, val_df):
        self.train_df = train_df
        self.val_df = val_df

    
    def train_val_scorer_df_maker(self, features, name, model=KNeighborsClassifier, params={}):
        '''Method parameters should include a list of features to be scored, a descriptive name for the column
        in the pandas DataFrame that is returned, an estimator for the GridSearchCV, and parameters for 
        the estimator. 
        
        The function returns a DataFrame that includes the train score and validation score from the GridSearch.'''
        
        self.results = {}
        self.features = features
        self.params = params
        self.model = model

        y_train = self.train_df['target']
        X_train = self.train_df[features]
        y_val = self.val_df['target']
        X_val = self.val_df[features]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        gridsearch = GridSearchCV(model(), param_grid=params, cv=5, n_jobs=-1)
        gridsearch.fit(X_train_scaled, y_train)
        
        self.best_params_ = gridsearch.best_params_

        train_score = gridsearch.score(X_train_scaled, y_train)
        val_score = gridsearch.score(X_val_scaled, y_val)

        self.results[name] = {'train_score':train_score, 'val_score':val_score}

        score_df = pd.DataFrame(self.results)
        return score_df


    def plot_top_down_feature_elimination_scores(self, features, model, params, random=False, reverse=False, noise=False):
        '''Method parameters require a list of features to be modeled and scored, as well as an estimator and 
        parameter grid for the GridSearchCV. From that list, a feature will be removed and the GridSearch refit
        with the new list. Train and validation scores are graphed, and points where the score decreased from the 
        previous iteration will be pointed out on the graph, until there is no more features remaining from the 
        input list. Each point will be labeled as the point that was removed before fitting and scoring.
        
        Options include: 
        random=  - if set to True, this method will randomize the input features list before plotting.
        reverse= - if set to True, this method will reverse the features list before running the method.
        noise=   - if set to True, the points plotted on the graph will be when the GridSearch score increases
                   after removal, instead of decreasing
        '''
        
        self.results = {}
        self.features = features
        self.params = params
        self.model = model
        
        # feature sorting.
        if random:
            shuffle(features)
            all_feats = features
            reverse_all_feats = list(reversed(all_feats))

        else:
            if reverse:
                all_feats = sorted(features, reverse=True)
                reverse_all_feats = sorted(features)
            else:
                all_feats = sorted(features)
                reverse_all_feats = sorted(features, reverse=True)

        self.results = {}

        #iteration through list of features, removing one feature after each iteration.
        for i in range(len(all_feats)):

            y_train = self.train_df['target']
            X_train = self.train_df[all_feats]
            y_val = self.val_df['target']
            X_val = self.val_df[all_feats]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # GridSearch fit and scoring with current iteration of feature list
            gridsearch = GridSearchCV(model(), param_grid=params, cv=5, n_jobs=-1)
            gridsearch.fit(X_train_scaled, y_train)
                        
            train_score = gridsearch.score(X_train_scaled, y_train)
            val_score = gridsearch.score(X_val_scaled, y_val)

            self.results[i] = {'train_score':train_score, 'val_score':val_score}
            
            # removing a feature from the input features list for the next iteration.
            all_feats = all_feats[:-1]

        results_df = pd.DataFrame(self.results).T

        
        # Plotting scores
        plt.figure(figsize=(10,10))
        plt.plot(results_df)
        plt.xlabel('Number of features removed from list')
        plt.ylabel('model score')
        _ = plt.xticks(range(len(features)))

        # customizing the title based on method parameters.
        if random:
            if noise:
                _ = plt.title('{}\nmodel scores from top-down feature removal (random, noise)'.format(model))
            else:
                _ = plt.title('{}\nmodel scores from top-down feature removal (random)'.format(model))
        else:
            if reverse:
                if noise:
                    _ = plt.title('{}\nmodel scores from top-down feature removal (reverse, noise)'.format(model))
                else: 
                    _ = plt.title('{}\nmodel scores from top-down feature removal (reverse, real_feats)'.format(model))
            else:
                if noise:
                    _ = plt.title('{}\nmodel scores from top-down feature removal (noise)'.format(model))
                else:
                    _ = plt.title('{}\nmodel scores from top-down feature removal (real_feats)'.format(model))

        select_feats = []
        
        # Plotting either noise or real features on the graph.
        for ind,x in enumerate(results_df['val_score']):

            if noise:
                if ind != 0 and results_df['val_score'][ind] > results_df['val_score'][ind-1]:
                    plt.plot(ind, results_df['val_score'][ind], 'o', color='b', label='feat_{}'.format(reverse_all_feats[ind-1]))
                    select_feats.append(reverse_all_feats[ind-1])
            else:
                if ind != 0 and results_df['val_score'][ind] < results_df['val_score'][ind-1]:
                    plt.plot(ind, results_df['val_score'][ind], 'o', color='r', label='feat_{}'.format(reverse_all_feats[ind-1]))
                    select_feats.append(reverse_all_feats[ind-1])

        _ = plt.legend()

        plt.show()

        return results_df, select_feats

    def list_top_dipped_feats(self, features, model, params, noise=False, random=False, n_feats_return=5):
        ''' This method runs the plot_top_down_feature_elimination_scores first with reverse=False, and then again
        with reverse=True. It then returns a list with the unique features from the aggregated list of features.
        
        Method parameters require a list of features to be modeled and scored, as well as an estimator and 
        parameter grid for the GridSearchCV.
        
        Options include: 
        random=  - if set to True, this method will randomize the input features list before plotting.
        noise=   - if set to True, the points plotted on the graph will be when the GridSearch score increases
                   after removal, instead of decreasing.
        '''
        
        self.results = {}
        self.features = features
        self.params = params
        self.model = model
        
        if random:
            if noise:
                results_df_1, select_feats_1 = self.plot_top_down_feature_elimination_scores(features, model, params, \
                                                                                           random=True, noise=True)
                results_df_2, select_feats_2 = self.plot_top_down_feature_elimination_scores(features, model, params, \
                                                                                           random=True, noise=True)
                print("TOP RANDOMIZED NOISE FEATURES:",\
                      sorted(list(set(select_feats_1[:n_feats_return]+select_feats_2[:n_feats_return]))))
            else:
                results_df_1, select_feats_1 = self.plot_top_down_feature_elimination_scores(features, model, params, \
                                                                                             random=True)
                results_df_2, select_feats_2 = self.plot_top_down_feature_elimination_scores(features, model, params, \
                                                                                             random=True)
                print("TOP RANDOMIZED REAL FEATURES:",\
                      sorted(list(set(select_feats_1[:n_feats_return]+select_feats_2[:n_feats_return]))))

            return sorted(list(set(select_feats_1[:n_feats_return]+select_feats_2[:n_feats_return])))


        else:
            if noise:
                results_df, select_feats = self.plot_top_down_feature_elimination_scores(features, model, params, noise=True)
                reverse_results_df, reverse_select_feats = self.plot_top_down_feature_elimination_scores(features, model, \
                                                                                                         params, reverse=True,\
                                                                                                         noise=True)
                print("TOP NOISE FEATURES:",\
                      sorted(list(set(select_feats[:n_feats_return]+reverse_select_feats[:n_feats_return]))))

            else:
                results_df, select_feats = self.plot_top_down_feature_elimination_scores(features, model, params)
                reverse_results_df, reverse_select_feats = self.plot_top_down_feature_elimination_scores(features, model, \
                                                                                                       params, reverse=True)
                print("TOP REAL FEATURES:",\
                      sorted(list(set(select_feats[:n_feats_return]+reverse_select_feats[:n_feats_return]))))

            return sorted(list(set(select_feats[:n_feats_return]+reverse_select_feats[:n_feats_return])))


    def rotator_score_df_generator(self, list_of_feats, model, params):
        '''This method takes a list of features, as well as a model and parameter grid for the GridSearchCV. 
        GridSearch is performed on all variations of the list with one feature removed. A pandas DataFrame
        is returned with the train and validation scores for every iteration.'''
        
        
        self.results = {}
        self.features = list_of_feats
        
        removed_feature_list = []
        rotater_list = list_of_feats

        for i in range(len(list_of_feats)):

            removed_feat = rotater_list[-1]
            feat_list = rotater_list[:-1]

            y_train = self.train_df['target']
            X_train = self.train_df[feat_list]
            y_val = self.val_df['target']
            X_val = self.val_df[feat_list]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            gridsearch = GridSearchCV(model(), param_grid=params, cv=5, n_jobs=-1)
            gridsearch.fit(X_train_scaled, y_train)

            train_score = gridsearch.score(X_train_scaled, y_train)
            val_score = gridsearch.score(X_val_scaled, y_val)

            self.results[removed_feat] = {'train_score':train_score, 'val_score':val_score, }

            rotater_list = [removed_feat] + feat_list

        rotater_df = pd.DataFrame(self.results).T
        return rotater_df

    def removerizer(self, full_features_list, features_to_remove):
        '''Takes in two lists, and returns the difference between them.'''
        
        new_list = full_features_list

        for x in features_to_remove:
            new_list.remove(x)

        return new_list

    def brute_force_feature_combination_score_generator(self, list_of_feats, model, params, n_feats=2):
        '''Method parameters should include a list of features to be fit and scored by GridSearchCV, as well as 
        an estimator and parameter grid for the GridSearch. 
        
        This method finds every combination of features, with each combination of features including n_feats'''
        
        self.features = list_of_feats
        self.model = model
        self.params = params
        
        list_of_combos = []

        # Making a list of every combination of features. 
        for combo in combinations(list_of_feats, n_feats):
            list_of_combos.append(list(combo))

        # Printing the number of combinations so that I know how long the process will take.
        print('Number of combinations:', len(list_of_combos))

        self.results = {}

        # fitting and scoring all combinations of features.
        for features in list_of_combos:

            y_train = self.train_df['target']
            X_train = self.train_df[features]
            y_val = self.val_df['target']
            X_val = self.val_df[features]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            gridsearch = GridSearchCV(model(), param_grid=params, cv=5, n_jobs=-1)
            gridsearch.fit(X_train_scaled, y_train)

            train_score = gridsearch.score(X_train_scaled, y_train)
            val_score = gridsearch.score(X_val_scaled, y_val)

            self.results['{}'.format(features)] = {'train_score':train_score, 'val_score':val_score, 'features':features}

        score_df = pd.DataFrame(self.results).T.sort_values('val_score', ascending=False)

        return score_df
    
    def make_set_from_list_of_lists(self, list_of_lists):
        
        uniques = []
        
        for l in list_of_lists:
            for x in l:
                uniques = uniques + [x]

        return list(set(uniques))
    
    

### These three functions are for using regression to identify colinearity in features to sort out noise features.

    def calculate_r2_for_feature(self, data, feature, model):
        new_data = data.drop(feature, axis=1)

        X_train, \
        X_test,  \
        y_train, \
        y_test = train_test_split(
            new_data,data[feature],test_size=0.25
        )

        if model == 'DecisionTreeRegressor':
            regressor = model()
            regressor.fit(X_train, y_train)
            score = regressor.score(X_test, y_test)

        else:    
            ss = StandardScaler()
            ss.fit(X_train, y_train)
            X_train_scaled = ss.transform(X_train)
            X_test_scaled = ss.transform(X_test)

            regressor = model()
            regressor.fit(X_train_scaled,y_train)

            score = regressor.score(X_test_scaled,y_test)

        return score


    def mean_r2_for_feature(self, data, feature, model):
        scores = []
        for _ in range(10):
            scores.append(self.calculate_r2_for_feature(data, feature, model))

        scores = np.array(scores)
        return scores.mean()


    def mean_r2_for_all_features(self, data, model):
        ''' 
        This method takes a pandas DataFrame and returns an r2 scores for every feature (column)
        in the DataFrame.
        '''
        feature_col_names = ['feat_{}'.format(i) for i in range(0,len(data.columns))]
        
        mean_scores = []
        for col in data.drop('target', axis=1).columns:
            mean_scores.append(self.mean_r2_for_feature(data.drop('target', axis=1), col, model))
        
        mean_score_df = pd.DataFrame([mean_scores, feature_col_names]).T.sort_values(0, ascending=False)
        mean_score_df.columns = ['r2_score','feature']
        
        return mean_score_df

    
# other functions
def con_cur_to_class_db():
    '''This function will return a connection and cursor object from the expanded Madelon data that was provided 
    by the instructors. It is not the downloaded version of the data from the UCI website.'''
    
    con = pg2.connect(host='34.211.227.227',
                  dbname='postgres',
                  user='postgres')
    cur = con.cursor(cursor_factory=RealDictCursor)
    
    return con, cur
    
def test_features_for_normal(df, features):
    
    ss = StandardScaler()
    
    normals = {}

    X = df[features]  
    X_sc = ss.fit_transform(X)
    
    for f in features:

        p = normaltest(X_sc[f]).pvalue
        
        if p >= 0.05:
            normals[f] = normaltest(X_sc[f]).pvalue
    
    return normals.keys()
