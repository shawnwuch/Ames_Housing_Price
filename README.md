# Ames_Housing_Price
In this reposity I performed an exploratory data analysis on the Ames Housing Dataset prepared by De Cock(2011), and then built several predictive models using Scikit-learn APIs. The dataset consists of 2930 observations of house sales documented in Ames between 2006-2010. Each property in the sale is described by 23 nominal, 23 ordinal, 14 discrete, and 20 continuous variables. The purpose of this exercise is to build an automatic valuation model (AVM) for predicting the price of a house in a sale. The AVM can help both the home buyers and sellers evaluate a property's market price more accurately and close the deal more efficiently.

I will use this historical dataset to demonstrate the basic steps for understanding a housing dataset and cleaning the dataset with imputation and feature transformation. The feature engineering steps for this particular housing price dataset include

1. Imputing missing values in the features sequentially (be aware of duplicates if there is any)
2. Transforming numerical variables into categorical variables (e.g., Mo.Sold, MS.SubClass etc.)
3. Label Encoding some categorical variables that may have information in their ordering
4. Box Cox / log transformation of skewed features
5. Normalization of numerical features to avoid ill-conditioning
6. Getting dummy variables for categorical features

The final dataset has a dimension of (2930, 248). 70% of the observations in the dataset is used to train predictive models built from Random Forest method, LASSO/Ridge/ElasticNet regression, and Gradient Boosting method. When applying the developed predictive models to the 30% observations intentionally left for the test set, we obtain a R^2 of 0.88-0.93 when comparing the predicted sale price of a house to the true sale price recorded in the dataset.

Reference: De Cock, D. (2011). Ames, Iowa: Alternative to the Boston housing data as an end of semester regression project. Journal of Statistics Education, 19(3).
