# Madelon Data: Feature Selection + Classification


### Data Description

This repository includes all the files related to my analysis of the Madelon synthetic dataset, which exemplifies the issues related to complex feature selection with big data.

[Madelon data description](https://archive.ics.uci.edu/ml/datasets/Madelon) from the UCI Website.

"Madelon is a two-class classification problem with continuous input variables. The difficulty is that the problem is multivariate and highly non-linear... Madelon is an artificial dataset containing data points grouped in 32 clusters placed on the vertices of a five dimensional hypercube and randomly labeled +1 or -1. The five dimensions constitute 5 informative features. 15 linear combinations of those features were added to form a set of 20 (redundant) informative features. Based on those 20 features one must separate the examples into the 2 classes (corresponding to the +-1 labels). We added a number of distractor feature called 'probes' having no predictive power. The order of the features and patterns were randomized." 


### Repository Contents:

1. **madelon_analysis_report.pdf** - This report describes the process and results for the entire data analysis. Start here.

1. **madelon_analyzer.py** - This was a class that I used to store all of the methods I used in the analysis. 
1. **Data** - This folder contains all of the pickles I made of model results throughout both Part 1 and Part 2 notebooks.

**Files relevant to Part 1:**
1. **web_madelon_data** - Folder containing the files from the UCI website.
1. **Madelon_UCI_Data_Analysis.ipynb** - This notebook shows the code and results from the Part 1 analysis
1. **UCI_madelon_dataframes_and_features_lists.py** - Contains a few lists that were the features found through different models in the UCI analysis.

**Files relevant to Part 2:**
1. **Expanded_madelon_data_10_percent_sample_analysis.ipynb** - This notebook shows the code and results for the analysis on a 10% sample of the 200,000 row dataset.
1. **Expanded_Data_Madelon_Analysis_small_samples.ipynb** - This notebook shows the code and results for the analysis on a very small sample of 3000 rows (out of 200,000). 
1. **Expanded_madelon_dataframes_and_features.py** - Contains a few lists that were the features found through different models in the expanded analysis.
1. **pickle_jar.py** - Since running the models on such a large dataset takes a long time, I stored the pickles in the Data folder and would run this script in the beginning of the Expanded_Data_Madelon_Analysis_small_samples.ipynb notebook so I would have all of the model results.