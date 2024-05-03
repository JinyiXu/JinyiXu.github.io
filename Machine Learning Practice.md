## Machine Learning Practice

Objective: After this assignment, you can build a pipeline that
1. Preprocesses realistic data (multiple variable types) in a pipeline that handles each variable type
1. Estimates a model using CV
1. Hypertunes a model on a CV folds within training sample
1. Finally, evaluate its performance in the test sample

Let's start by loading the data


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# load data and split off X and y
housing = pd.read_csv('input_data2/housing_train.csv')
y = np.log(housing.v_SalePrice)
housing = housing.drop('v_SalePrice',axis=1)
```

To ensure you can be graded accurately, we need to make the "randomness" predictable. (I.e. you should get the exact same answers every single time we run this.)

Per the recommendations in the [sk-learn documentation](https://scikit-learn.org/stable/common_pitfalls.html#general-recommendations), what that means is we need to put `random_state=rng` inside every function in this file that accepts "random_state" as an argument.



```python
# create test set for use later - notice the (random_state=rng)
rng = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(housing, y, random_state=rng)
```

## Part 1: Preprocessing the data

1. Set up a single pipeline called `preproc_pipe` to preprocess the data.
    1. For **all** numerical variables, impute missing values with SimpleImputer and scale them with StandardScaler
    1. `v_Lot_Config`: Use OneHotEncoder on it 
    1. Drop any other variables (handle this **inside** the pipeline)
1. Use this pipeline to preprocess X_train. 
    1. Describe the resulting data **with two digits.**
    1. How many columns are in this object?

_HINTS:_
- _You do NOT need to type the names of all variables. There is a lil trick to catch all the variables._
- _The first few rows of my print out look like this:_

| | count | mean | std | min  | 25%  | 50% |  75% |  max
| --- | --- | --- | ---  | ---  | --- |  --- |  --- |  ---
|  v_MS_SubClass | 1455 | 0 | 1 | -0.89 | -0.89 | -0.2 | 0.26 | 3.03
|  v_Lot_Frontage | 1455 | 0 | 1 | -2.2 | -0.43 | 0 | 0.39 | 11.07
|  v_Lot_Area  | 1455 | 0 | 1 | -1.17 | -0.39 | -0.11 | 0.19 | 20.68
| v_Overall_Qual | 1455 | 0 | 1 | -3.7 | -0.81 | -0.09 | 0.64 | 2.8


```python
# Identify numerical and categorical columns
numerical_cols = housing.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = ['v_Lot_Config']  # Only one categorical column to encode

# Create the preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a single ColumnTransformer using make_column_selector
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop' # Drops all other columns not specified
)

# Complete preprocessing pipeline
preproc_pipe = Pipeline(steps=[('preprocessor', preprocessor)])

# Apply the preprocessing pipeline to the training data
X_train_processed = preproc_pipe.fit_transform(X_train)

# Get the one-hot encoded column names and combine them with numerical columns
one_hot_feature_names = preproc_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['v_Lot_Config'])
transformed_columns = X_train.select_dtypes(include=np.number).columns.tolist() + one_hot_feature_names.tolist()

# Convert the processed data back to a DataFrame
X_train_df = pd.DataFrame(X_train_processed, columns=transformed_columns)

# Describing the data and transposing
X_train_desc = X_train_df.describe().round(2).transpose()

# Output the description and the number of processed columns
print(X_train_desc)
print(X_train_desc.shape[1]) # 8 columns
```

                           count  mean   std   min   25%   50%   75%    max
    v_MS_SubClass         1455.0  0.00  1.00 -0.89 -0.89 -0.20  0.26   3.03
    v_Lot_Frontage        1455.0  0.00  1.00 -2.19 -0.42 -0.05  0.40  11.08
    v_Lot_Area            1455.0  0.00  1.00 -1.17 -0.39 -0.11  0.19  20.68
    v_Overall_Qual        1455.0  0.00  1.00 -3.70 -0.81 -0.09  0.64   2.80
    v_Overall_Cond        1455.0  0.00  1.00 -4.30 -0.53 -0.53  0.41   3.24
    v_Year_Built          1455.0 -0.00  1.00 -3.08 -0.62  0.05  0.98   1.22
    v_Year_Remod/Add      1455.0  0.00  1.00 -1.63 -0.91  0.43  0.96   1.20
    v_Mas_Vnr_Area        1455.0 -0.00  1.00 -0.56 -0.56 -0.56  0.33   7.87
    v_BsmtFin_SF_1        1455.0  0.00  1.00 -0.96 -0.96 -0.16  0.65  11.20
    v_BsmtFin_SF_2        1455.0 -0.00  1.00 -0.29 -0.29 -0.29 -0.29   8.29
    v_Bsmt_Unf_SF         1455.0  0.00  1.00 -1.28 -0.77 -0.23  0.55   3.58
    v_Total_Bsmt_SF       1455.0  0.00  1.00 -2.39 -0.59 -0.14  0.55  11.35
    v_1st_Flr_SF          1455.0 -0.00  1.00 -2.07 -0.68 -0.19  0.55   9.76
    v_2nd_Flr_SF          1455.0 -0.00  1.00 -0.78 -0.78 -0.78  0.85   3.98
    v_Low_Qual_Fin_SF     1455.0  0.00  1.00 -0.09 -0.09 -0.09 -0.09  14.09
    v_Gr_Liv_Area         1455.0 -0.00  1.00 -2.23 -0.72 -0.14  0.43   7.82
    v_Bsmt_Full_Bath      1455.0 -0.00  1.00 -0.82 -0.82 -0.82  1.11   3.04
    v_Bsmt_Half_Bath      1455.0  0.00  1.00 -0.24 -0.24 -0.24 -0.24   7.94
    v_Full_Bath           1455.0  0.00  1.00 -2.84 -1.03  0.78  0.78   2.59
    v_Half_Bath           1455.0  0.00  1.00 -0.76 -0.76 -0.76  1.25   3.26
    v_Bedroom_AbvGr       1455.0  0.00  1.00 -3.51 -1.07  0.15  0.15   6.24
    v_Kitchen_AbvGr       1455.0 -0.00  1.00 -5.17 -0.19 -0.19 -0.19   4.78
    v_TotRms_AbvGrd       1455.0 -0.00  1.00 -2.83 -0.93 -0.30  0.33   5.39
    v_Fireplaces          1455.0 -0.00  1.00 -0.94 -0.94  0.63  0.63   5.32
    v_Garage_Yr_Blt       1455.0 -0.00  1.00 -3.41 -0.67  0.07  0.97   1.21
    v_Garage_Cars         1455.0 -0.00  1.00 -2.34 -1.03  0.28  0.28   2.91
    v_Garage_Area         1455.0 -0.00  1.00 -2.20 -0.69  0.01  0.46   4.65
    v_Wood_Deck_SF        1455.0  0.00  1.00 -0.74 -0.74 -0.74  0.59  10.54
    v_Open_Porch_SF       1455.0 -0.00  1.00 -0.71 -0.71 -0.31  0.32   7.67
    v_Enclosed_Porch      1455.0 -0.00  1.00 -0.36 -0.36 -0.36 -0.36   9.33
    v_3Ssn_Porch          1455.0  0.00  1.00 -0.09 -0.09 -0.09 -0.09  19.74
    v_Screen_Porch        1455.0 -0.00  1.00 -0.29 -0.29 -0.29 -0.29   9.69
    v_Pool_Area           1455.0 -0.00  1.00 -0.08 -0.08 -0.08 -0.08  17.05
    v_Misc_Val            1455.0 -0.00  1.00 -0.09 -0.09 -0.09 -0.09  24.19
    v_Mo_Sold             1455.0 -0.00  1.00 -2.03 -0.55 -0.18  0.56   2.04
    v_Yr_Sold             1455.0 -0.00  1.00 -1.24 -1.24  0.00  1.25   1.25
    v_Lot_Config_Corner   1455.0  0.18  0.38  0.00  0.00  0.00  0.00   1.00
    v_Lot_Config_CulDSac  1455.0  0.06  0.24  0.00  0.00  0.00  0.00   1.00
    v_Lot_Config_FR2      1455.0  0.02  0.15  0.00  0.00  0.00  0.00   1.00
    v_Lot_Config_FR3      1455.0  0.01  0.08  0.00  0.00  0.00  0.00   1.00
    v_Lot_Config_Inside   1455.0  0.73  0.44  0.00  0.00  1.00  1.00   1.00
    8


## Part 2: Estimating one model

_Note: A Lasso model is basically OLS, but it pushes some coefficients to zero. Read more in the `sklearn` User Guide._

1. Report the mean test score (**show 5 digits**) when you use cross validation on a Lasso Model (after using the preprocessor from Part 1) with
    - alpha = 0.3, 
    - CV uses 10 `KFold`s
    - R$^2$ scoring 
1. Now, still using CV with 10 `KFold`s and R$^2$ scoring, let's find the optimal alpha for the lasso model. You should optimize the alpha out to the exact fifth digit that yields the highest R2. 
    1. According to the CV function, what alpha leads to the highest _average_ R2 across the validation/test folds? (**Show 5 digits.**)
    1. What is the mean test score in the CV output for that alpha?  (**Show 5 digits.**)
    1. After fitting your optimal model on **all** of X_train, how many of the variables did it select? (Meaning: How many coefficients aren't zero?)
    3. After fitting your optimal model on **all** of X_train, report the 5 highest  _non-zero_ coefficients (Show the names of the variables and the value of the coefficients.)
    4. After fitting your optimal model on **all** of X_train, report the 5 lowest _non-zero_ coefficients (Show the names of the variables and the value of the coefficients.)
    5. After fitting your optimal model on **all** of X_train, now use your predicted coefficients on the test ("holdout") set! What's the R2?


```python
#1
# Creating a pipeline that includes preprocessing and the Lasso model
lasso_model = Pipeline(steps=[
    ('preprocessor', preproc_pipe),
    ('lasso', Lasso(alpha=0.3))
])

# Setting up cross-validation with 10 folds
cv = KFold(n_splits=10, shuffle=True, random_state=0)

# Cross-validate the Lasso model using R^2 as the scoring metric
scores = cross_val_score(lasso_model, X_train, y_train, cv=cv, scoring='r2')

# Reporting the mean test score with 5 digits
mean_test_score = scores.mean().round(5)

mean_test_score
```




    0.08344




```python
#2
if X_train.shape[0] >= 10:
    from sklearn.model_selection import GridSearchCV
    alphas_to_test = np.logspace(-4, 1, 100)  # Define a range of alpha values to test
    
    # Grid search to find the optimal alpha
    grid_search = GridSearchCV(lasso_model, {'lasso__alpha': alphas_to_test}, cv=cv, scoring='r2')
    grid_search.fit(X_train, y_train)
    best_alpha = grid_search.best_params_['lasso__alpha'].round(5)
    best_cv_score = grid_search.best_score_.round(5)
    print("Best alpha:", best_alpha)
    print("Best CV score:", best_cv_score)
    
optimal_lasso = Lasso(alpha=best_alpha)

# Create and fit the optimal Lasso model pipeline
optimal_lasso_model = Pipeline([
    ('preprocessor', preprocessor),  # Using the predefined preprocessor from Part 1
    ('lasso', optimal_lasso)
])

optimal_lasso_model.fit(X_train, y_train)

# Retrieve and count non-zero coefficients
coefficients = optimal_lasso_model.named_steps['lasso'].coef_
non_zero_coefficients = np.count_nonzero(coefficients)

# Identify the names of features from the preprocessor, assuming 'transformed_columns' are defined
feature_names = transformed_columns  # This should be set after preprocessing pipeline

# Sorting coefficients and corresponding feature names
coef_feature_pairs = sorted(zip(coefficients, feature_names), key=lambda x: abs(x[0]), reverse=True)

# Find the 5 highest non-zero coefficients
highest_coefs = [pair for pair in coef_feature_pairs if pair[0] != 0][:5]

# Find the 5 lowest non-zero coefficients (those closest to zero)
lowest_coefs = [pair for pair in coef_feature_pairs if pair[0] != 0][-5:]

# Evaluate the model on the test set
test_r2 = optimal_lasso_model.score(X_test, y_test)

print(f"non_zero_coefficients: {non_zero_coefficients}")
print(f"highest_coefs: {highest_coefs}")
print(f"lowest_coefs: {lowest_coefs}")
print(f"test_r2: {test_r2}")

```

    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.186e-02, tolerance: 2.049e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.652e-02, tolerance: 2.051e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.567e-02, tolerance: 2.041e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.280e-02, tolerance: 2.047e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.793e-02, tolerance: 2.053e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.986e-02, tolerance: 2.013e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.111e-02, tolerance: 2.059e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.231e-02, tolerance: 2.049e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.685e-02, tolerance: 2.051e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.599e-02, tolerance: 2.041e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.321e-02, tolerance: 2.047e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.828e-02, tolerance: 2.053e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.021e-02, tolerance: 2.013e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.130e-02, tolerance: 2.059e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.268e-02, tolerance: 2.049e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.712e-02, tolerance: 2.051e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.624e-02, tolerance: 2.041e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.356e-02, tolerance: 2.047e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.856e-02, tolerance: 2.053e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.050e-02, tolerance: 2.013e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.139e-02, tolerance: 2.059e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.295e-02, tolerance: 2.049e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.728e-02, tolerance: 2.051e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.638e-02, tolerance: 2.041e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.381e-02, tolerance: 2.047e-02
      model = cd_fast.enet_coordinate_descent(
    /Users/xujinyi/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:628: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.307e-02, tolerance: 2.049e-02
      model = cd_fast.enet_coordinate_descent(


    Best alpha: 0.00739
    Best CV score: 0.83583
    non_zero_coefficients: 21
    highest_coefs: [(0.1342087697103703, 'v_Overall_Qual'), (0.09833040021589204, 'v_Gr_Liv_Area'), (0.0667940687147935, 'v_Year_Built'), (0.04754035558480722, 'v_Garage_Cars'), (0.036340984770179816, 'v_Overall_Cond')]
    lowest_coefs: [(0.005480491491301416, 'v_1st_Flr_SF'), (-0.004380205216281666, 'v_Kitchen_AbvGr'), (0.0043170673205364715, 'v_Bedroom_AbvGr'), (0.004202854856695856, 'v_BsmtFin_SF_1'), (-0.0028839423803625366, 'v_Pool_Area')]
    test_r2: 0.8659218891568778


## Part 3: Optimizing and estimating your own model

You can walk! Let's try to run! The next skill level is trying more models and picking your favorite. 

Read this whole section before starting!  

1. Output 1: Build a pipeline with these 3 steps and **display the pipeline** 
    1. step 1: preprocessing: possible preprocessing things you can try include imputation, scaling numerics, outlier handling, encoding categoricals, and feature creation (polynomial transformations / interactions) 
    1. step 2: feature "selection": [Either selectKbest, RFEcv, or PCA](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection)
    1. step 3: model estimation: [e.g. a linear model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) or an [ensemble model like HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)
1. Pick two hyperparameters to optimize: Both of the hyperparameters you optimize should be **numeric** in nature (i.e. not median vs mean in the imputation step). Pick parameters you think will matter the most to improve predictions. 
    - Put those parameters into a grid search and run it. 
1. Output 2: Describe what each of the two parameters you chose is/does, and why you thought it was important/useful to optimize it.
1. Output 3: Plot the average (on the y-axis) and STD (on the x-axis) of the CV test scores from your grid search for 25+ models you've considered. Highlight in red the dot corresponding to the model **you** prefer from this set of options, and **in the figure somewhere,** list the parameters that red dot's model uses. 
    - Your plot should show at least 25 _**total**_ combinations.
    - You'll try far more than 25 combinations to find your preferred model. You don't need to report them all.
1. Output 4: Tell us the set of possible values for each parameter that were reported in the last figure.
    - For example: "Param 1 could be 0.1, 0.2, 0.3, 0.4, and 0.5. Param 2 could be 0.1, 0.2, 0.3, 0.4, and 0.5." Note: Use the name of the parameter in your write up, don't call it "Param 1".
    - Adjust your gridsearch as needed so that your preferred model doesn't use a hyperparameter whose value is the lowest or highest possible value for that parameter. Meaning: If the optimal is at the high or low end for a parameter, you've _probably_ not optimized it!
1. Output 5: Fit your pipeline on all of X_train using the optimal parameters you just found. Now use your predicted coefficients on the test ("holdout") set! **What's the R2 in the holdout sample with your optimized pipeline?**

- I picked 'k' in 'SelectKBest' and 'max_leaf_nodes' in 'HistGradientBoostingRegressor' as two parameters. 
- 'k' determines the number of top features to keep based on their statistical relationship with the target variable. Adjusting k allows me to examine how the number of features affects the performance, letting you choose a model that is both efficient and effective.
- 'max_leaf_nodes' controls the maximum number of leaves a tree can have in the boosting process. In tree-based models, each node represents a condition under which the dataset is split based on a certain feature. By tuning this parameter, I can find a good trade-off between the model being too complex or too simplistic, aiming to enhance generalization to new, unseen data.
- These parameters are chosen because they directly affect the fundamental aspects of the modeling process: feature handling and model structure. Optimizing these can lead to noticeable improvements in model accuracy, training efficiency, and the ability to generalize from training data to real-world application scenarios.


```python
# Define the preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), make_column_selector(dtype_include=np.number))
])

# Complete pipeline with preprocessing, feature selection, and model estimation
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', SelectKBest()),
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

# Expanded parameter grid for more combinations
param_grid = {
    'selector__k': np.linspace(1, 20, 10, dtype=int),  # More steps in feature selection
    'regressor__max_leaf_nodes': np.linspace(10, 100, 10, dtype=int)  # More steps in max_leaf_nodes
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', return_train_score=True)
grid_search.fit(X_train, y_train)

# Results plotting
results = pd.DataFrame(grid_search.cv_results_)
means = results['mean_test_score']
stds = results['std_test_score']

plt.figure(figsize=(10, 6))
plt.errorbar(stds, means, fmt='o', color='blue', ecolor='lightgray', elinewidth=3, capsize=0)
plt.xlabel('Standard Deviation of CV Scores')
plt.ylabel('Mean CV Score (R2)')
plt.title('Grid Search Scores')

# Highlight the best model
best_index = np.argmax(means)
plt.scatter(stds[best_index], means[best_index], color='red', s=100)  # Red dot
plt.annotate(f"Best Model\nk={grid_search.best_params_['selector__k']}\nMax Leafs={grid_search.best_params_['regressor__max_leaf_nodes']}",
             (stds[best_index], means[best_index]), textcoords="offset points", xytext=(0,10), ha='center')

plt.grid(True)
plt.show()
print(f'Feature selection k values tested: {param_grid["selector__k"]}')
print(f'Max leaf nodes values tested: {param_grid["regressor__max_leaf_nodes"]}')


# Final model training and testing
final_model = grid_search.best_estimator_
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
final_r2 = r2_score(y_test, y_pred)
print(f'R2 score on the test set: {final_r2}')
```


    
![png](output_11_0.png)
    


    Feature selection k values tested: [ 1  3  5  7  9 11 13 15 17 20]
    Max leaf nodes values tested: [ 10  20  30  40  50  60  70  80  90 100]
    R2 score on the test set: 0.8696260356418277

