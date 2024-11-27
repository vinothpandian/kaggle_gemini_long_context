# Scale and Transform
# Normalize
Normalization is a technique often applied as part of data preparation for machine learning. The goal of normalization is to rescale the values of numeric columns in the dataset without distorting differences in the ranges of values or losing information. There are several methods available for normalization, by default, PyCaret uses zscore.

PARAMETERS
normalize: bool, default = False
When set to True, the feature space is transformed using the method defined under the normalized_method parameter. 

normalize_method: string, default = ‘zscore’
Defines the method to be used for normalization. By default, the method is set to zscore. The other available options are:

z-score
 The standard zscore is calculated as z = (x – u) / s

minmax
 scales and translates each feature individually such that it is in the range of 0 – 1.

maxabs
 scales and translates each feature individually such that the maximal absolute value of each feature will be 1.0. It does not shift/center the data and thus does not destroy any sparsity.

robust
 scales and translates each feature according to the Interquartile range. When the dataset contains outliers, the robust scaler often gives better results.

 # Feature Transform
While normalization rescales the data within new limits to reduce the impact of magnitude in the variance, Feature transformation is a more radical technique. Transformation changes the shape of the distribution such that the transformed data can be represented by a normal or approximate normal distribution. There are two methods available for transformation yeo-johnson and quantile.

PARAMETERS
transformation: bool, default = False
When set to True, a power transformer is applied to make the data more normal / Gaussian-like. This is useful for modeling issues related to heteroscedasticity or other situations where normality is desired. The optimal parameter for stabilizing variance and minimizing skewness is estimated through maximum likelihood.

transformation_method: string, default = ‘yeo-johnson’
Defines the method for transformation. By default, the transformation method is set to yeo-johnson. The other available option is quantile transformation. Both the transformation transforms the feature set to follow a Gaussian-like or normal distribution. Quantile transformer is non-linear and may distort linear correlations between variables measured at the same scale.

# Target Transform
Target Transformation is similar to Feature Transformation as it will change the shape of the distribution of the target variable instead of Features. This feature is only available in pycaret.regression module.

PARAMETERS
transform_target: bool, default = False
When set to True, target variable is transformed using the method defined in transform_target_method parameter. Target transformation is applied separately from feature transformations.

transform_target_method: string, default = ‘yeo-johnson’
Defines the method for transformation. By default, the transformation method is set to yeo-johnson. The other available option for transformation is quantile. Ignored when transform_target = False.

