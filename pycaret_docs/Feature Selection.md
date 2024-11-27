# Feature Selection
# Feature Selection
Feature Importance is a process used to select features in the dataset that contribute the most in predicting the target variable. Working with selected features instead of all the features reduces the risk of over-fitting, improves accuracy, and decreases the training time. In PyCaret, this can be achieved using feature_selection parameter. 

PARAMETERS
feature_selection: bool, default = False
When set to True, a subset of features is selected based on a feature importance score determined by feature_selection_estimator.

feature_selection_method: str, default = 'classic'

Algorithm for feature selection. Choose from:

'univariate': Uses sklearn's SelectKBest.

'classic': Uses sklearn's SelectFromModel.

'sequential': Uses sklearn's SequentialFeatureSelector.

feature_selection_estimator: str or sklearn estimator, default = 'lightgbm'

Classifier used to determine the feature importance. The estimator should have a feature_importances_ or coef_ attribute after fitting. If None, it uses LGBClassifier. This parameter is ignored when feature_selection_method=univariate.

n_features_to_select: int or float, default = 0.2

The maximum number of features to select with feature_selection. If <1, it's the fraction of starting features. Note that this parameter doesn't take features in ignore_features or keep_features into account when counting.

# Remove Multicollinearity
Multicollinearity (also called collinearity) is a phenomenon in which one feature variable in the dataset is highly linearly correlated with another feature variable in the same dataset. Multicollinearity increases the variance of the coefficients, thus making them unstable and noisy for linear models. One such way to deal with Multicollinearity is to drop one of the two features that are highly correlated with each other. This can be achieved in PyCaret using remove_multicollinearity parameter.

PARAMETERS
remove_multicollinearity: bool, default = False
When set to True, features with the inter-correlations higher than the defined threshold are removed. For each group, it removes all except the feature with the highest correlation to y.

multicollinearity_threshold: float, default = 0.9
Minimum absolute Pearson correlation to identify correlated features. The default value removes equal columns. Ignored when remove_multicollinearity is not True.

# Principal Component Analysis
Principal Component Analysis (PCA) is an unsupervised technique used in machine learning to reduce the dimensionality of a data. It does so by compressing the feature space by identifying a subspace that captures most of the information in the complete feature matrix. It projects the original feature space into lower dimensionality. 

PARAMETERS

pca: bool, default = False
When set to True, dimensionality reduction is applied to project the data into a lower dimensional space using the method defined in pca_method parameter.

pca_method: string, default = ‘linear’
Method with which to apply PCA. Possible values are:

'linear': Uses Singular Value Decomposition.

'kernel': Dimensionality reduction through the use of RBF kernel.

'incremental': Similar to 'linear', but more efficient for large datasets.

pca_components: int/float, default = 0.99
Number of components to keep. if pca_components is a float, it is treated as a target percentage for information retention. When pca_components is an integer it is treated as the number of features to be kept. pca_components must be strictly less than the original number of features in the dataset.

pca_components: int, float, str or None, default = None 
Number of components to keep. This parameter is ignored when pca=False. 

If None: All components are kept. 

If int: Absolute number of components. - 

If float: Such an amount that the variance that needs to be explained is greater than the percentage specified by n_components. Value should lie between 0 and 1 (ony for pca_method='linear'). 

If 'mle': Minka’s MLE is used to guess the dimension (ony for pca_method='linear').

# Ignore Low Variance
Sometimes a dataset may have a categorical feature with multiple levels, where distribution of such levels are skewed and one level may dominate over other levels. This means there is not much variation in the information provided by such feature.  For a ML model, such feature may not add a lot of information and thus can be ignored for modeling. This can be achieved in PyCaret using low_variance_threshold parameter.

PARAMETERS
low_variance_threshold: float or None, default = None

Remove features with a training-set variance lower than the provided threshold. If 0, keep all features with non-zero variance, i.e. remove the features that have the same value in all samples. If None, skip this transformation step.

# 