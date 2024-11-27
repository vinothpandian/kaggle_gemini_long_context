# Feature Engineering
# Polynomial Features
In machine learning experiments, the relationship between the dependent and independent variables is often assumed to be linear; however, this is not always the case. Sometimes the relationship between dependent and independent variables is more complex. Creating new polynomial features sometimes might help in capturing that relationship, which otherwise may go unnoticed. 

PARAMETERS
polynomial_features: bool, default = False
When set to True, new features are created based on all polynomial combinations that exist within the numeric features in a dataset to the degree defined in the polynomial_degree parameter.

polynomial_degree: int, default = 2
Degree of polynomial features. For example, if an input sample is two dimensional and of the form [a, b], the polynomial features with degree = 2 are: [1, a, b, a^2, ab, b^2].

# Group Features
When dataset contains features that are related to each other in someway, for example: features recorded at some fixed time intervals, then new statistical features such as mean, median, variance and standard deviation for a group of such features can be created from existing features using group_features parameter.

PARAMETERS
group_features: list or list of list, default = None
When a dataset contains features that have related characteristics, the group_features param can be used for statistical feature extraction. For example, if a dataset has numeric features that are related with each other (i.e ‘Col1’, ‘Col2’, ‘Col3’), a list containing the column names can be passed under group_features to extract statistical information such as the mean, median, mode and standard deviation.

group_names: list, default = None
When group_features is passed, a name of the group can be passed into the group_names parameter as a list containing strings. The length of a group_names list must equal to the length of group_features. When the length doesn’t match or the name is not passed, new features are sequentially named such as group_1, group_2 etc.

# Bin Numeric Features
Feature binning is a method of turning continuous variables into categorical values using pre-defined number of bins. It is effective when a continuous feature has too many unique values or few extreme values outside the expected range. Such extreme values influence on the trained model, thereby affecting the prediction accuracy of the model. In PyCaret, continuous numeric features can be binned into intervals using bin_numeric_features parameter. PyCaret uses the ‘sturges’ rule to determine the number of bins and uses K-Means clustering to convert continuous numeric features into categorical features.

PARAMETERS
bin_numeric_features: list, default = None
When a list of numeric features is passed they are transformed into categorical features using K-Means, where values in each bin have the same nearest center of a 1D k-means cluster. The number of clusters are determined based on the 'sturges' method. It is only optimal for gaussian data and underestimates the number of bins for large non-gaussian datasets.

# Combine Rare Levels
Sometimes a dataset can have a categorical feature (or multiple categorical features) that has a very high number of levels (i.e. high cardinality features). If such feature (or features) are encoded into numeric values, then the resultant matrix is a sparse matrix. This not only makes experiment slow due to manifold increment in the number of features and hence the size of the dataset, but also introduces noise in the experiment. Sparse matrix can be avoided by combining the rare levels in the feature(or features) having high cardinality. This can be achieved in PyCaret using rare_to_value parameter.

PARAMETERS
rare_to_value: float or None, default=None

Minimum fraction of category occurrences in a categorical column. If a category is less frequent than rare_to_value * len(X), it is replaced with the string in rare_value. Use this parameter to group rare categories before encoding the column. If None, ignores this step.

rare_value: str, default="rare"

Value with which to replace rare categories. Ignored when rare_to_value is None
