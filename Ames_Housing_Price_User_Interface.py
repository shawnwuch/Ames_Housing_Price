#Begin by setting up the Python environment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  # Matlab-style plotting
plt.rcParams.update({'font.size': 22})

import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import norm, skew, boxcox_normmax
from scipy.special import boxcox1p
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
#from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import robust_scale
from sklearn.metrics import mean_squared_error

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

from subprocess import check_output
import sys
import pickle

#print(check_output(["ls", "./all"]).decode("utf8")) #check the files available in the directory


def main():
		# This basic command line argument parsing code is provided.
		# Make a list of command line arguments, omitting the [0] element
		# which is the script itself.
		args = sys.argv[1:]
		read_count = 0
		
		if not args:
			print("Usage: python Amne_Interview.py ./path/to/training_set.csv ./path/to/test_set.csv")  
			sys.exit(1)

		for filename in args:	
			print("Reading", filename, "...")
			read_count += 1
			
			if read_count == 1:
				#Import the datasets in pandas dataframe
				df_full = pd.read_csv(filename)
				df_full.drop(['Order', 'PID'], axis=1, inplace=True)
				
				##print("There are ", len(df_full.dtypes[df_full.dtypes != "object"].index), "numeric features.")
				#corrmat = df_full.corr()
				#f, ax = plt.subplots(figsize=(12, 12))
				#sns.heatmap(corrmat, square=True);
				#
				#
				#SalePriceCorr = df_full.corr().abs()['SalePrice']
				##print(SalePriceCorr.sort_values(kind="quicksort", ascending=False))
				#
				#
				##bivariate analysis of SalePrice and continuous variables
				#correlated_features = ('Gr.Liv.Area', 'Total.Bsmt.SF', 'Year.Built')
				#for feature in correlated_features:
				#    data = pd.concat([df_full['SalePrice'], df_full[feature]], axis=1)
				#    data.plot.scatter(x=feature, y='SalePrice', ylim=(0,800000));
				#
				#
				## box plots of SalePrice and ordinal discrete variables
				#correlated_features = ('Overall.Qual', 'Garage.Cars', 'Full.Bath')
				#for feature in correlated_features:
				#    data = pd.concat([df_full['SalePrice'], df_full[feature]], axis=1)
				#    f, ax = plt.subplots(figsize=(8, 6))
				#    fig = sns.boxplot(x=feature, y="SalePrice", data=data)
				#    fig.axis(ymin=0, ymax=800000);
				#
				#
				## SalePrice is the response variable we need to predict. Do some analysis on it first.
				#sns.distplot(df_full['SalePrice'] , fit=norm);
				#
				## Get the fitted parameters used by the function
				#(mu, sigma) = norm.fit(df_full['SalePrice'])
				#
				##Now plot the distribution
				#plt.legend(['Normal dist. \n$\mu=$ {:.2f} \n$\sigma=$ {:.2f}'.format(mu, sigma)], loc='best')
				#plt.ylabel('PDF')
				#plt.title('SalePrice distribution')
				#
				##Create the Normal Probability Plot to check if SalePrice is normally distributed
				#fig = plt.figure()
				#res = stats.probplot(df_full['SalePrice'], plot=plt)
				#plt.show()
				
					
				# Apply a Box-Cox (or log) tranformation to make the SalePrice more normally distributed. It will help later on when training our (linear) models.
				# *Reference for Box-Cox transformation: https://www.itl.nist.gov/div898/handbook/eda/section3/boxcoxno.htm*
					
					
				# df_full["SalePrice"] = np.log1p(df_full["SalePrice"])
				lamda = boxcox_normmax(df_full["SalePrice"]) # the lamda factor
				df_full["SalePrice"] = boxcox1p(df_full["SalePrice"], lamda)
					
				## SalePrice is the response variable we need to predict. Do some analysis on it first.
				#sns.distplot(df_full["SalePrice"] , fit=norm);
				#
				## Get the fitted parameters used by the function
				#(mu, sigma) = norm.fit(df_full["SalePrice"])
				#
				##Now plot the distribution
				#plt.legend(['Normal dist. \n$\mu=$ {:.2f} \n$\sigma=$ {:.2f}'.format(mu, sigma)], loc='best')
				#plt.ylabel('PDF')
				#plt.title('SalePrice distribution')
				#
				##Create the Normal Probability Plot to check if SalePrice is normally distributed
				#fig = plt.figure()
				#res = stats.probplot(df_full["SalePrice"], plot=plt)
				#plt.show()
					
					
				# Separate the Response (SalePrice) from Features
				Response = df_full["SalePrice"]
				df_full.drop(["SalePrice"], axis=1, inplace=True) 
					
					
				# ## Feature Engineering
				# 1. Imputing missing values in the features sequentially (be aware of duplicates if there is any)
				# 2. Transforming numerical variables into categorical variables (e.g., Mo.Sold, MS.SubClass etc.)
				# 3. Label Encoding some categorical variables that may have information in their ordering
				# 4. Box Cox / log transformation of skewed features
				# 5. Normalization of numerical features to avoid ill-conditioning
				# 6. Getting dummy variables for categorical features
					
					
				df_full_na = (df_full.isnull().sum() / len(df_full)) * 100  # calculate the ratio of missing data
				df_full_na = df_full_na.drop(df_full_na[df_full_na == 0].index).sort_values(ascending=False) # drop features w/o any missing data
				missing_data = pd.DataFrame({'Missing Ratio' :df_full_na})
					
					
				## 1. Imputing missing values in the features sequentially
				# * Pool.QC: NA means "No Pool".
				# * Misc.Feature: NA means "no misc feature".
				# * Alley: NA means "no alley access".
				# * Fence: NA means "no fence".
				# * Fireplace.Qu : NA means "no fireplace".
				# * Lot.Frontage : Assuming that the lot in front of a house does not vary by much within one neighborhood, I will b.
				# * Garage.Yr.Blt, Garage.Area and Garage.Cars : Replacing missing data with 0 for houses with no garage.
				# * Garage.Type, Garage.Finish, Garage.Qual and Garage.Cond : NA means "no garage".
				# * Bsmt.Qual, Bsmt.Cond, Bsmt.Exposure, Bsmt.Fin.Type1 and Bsmt.Fin.Type.2 : For these categorical features NA means that there is no basement.
				# * BsmtFin.SF.1, BsmtFin.SF.2, Bsmt.Unf.SF, Total.Bsmt.SF, Bsmt.Full.Bath and Bsmt.Half.Bath : Missing values are likely 0 for these numeric features of a house without a basement.
				# * Mas.Vnr.Area and Mas.Vnr.Type : NA means no masonry veneer for the houses, so we will fill 0 for the area and None for the type.
				# * Electrical : This categorical feature has one missing value. We will fill it in with the mode (SBrkr:	Standard Circuit Breakers & Romex).
					
					
				df_full["Pool.QC"] = df_full["Pool.QC"].fillna("None")
					
				# double check if all Nan are dealt
				#print(df_full["Pool.QC"].isnull().sum()) 
				
				
				df_full["Misc.Feature"] = df_full["Misc.Feature"].fillna("None")
				df_full["Alley"] = df_full["Alley"].fillna("None")
				df_full["Fence"] = df_full["Fence"].fillna("None")
				df_full["Fireplace.Qu"] = df_full["Fireplace.Qu"].fillna("None")
				
				# double check if all Nan are dealt
				#print(df_full["Misc.Feature"].isnull().sum())
				#print(df_full["Alley"].isnull().sum())
				#print(df_full["Fence"].isnull().sum())
				#print(df_full["Fireplace.Qu"].isnull().sum())
					
					
				#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
				df_full["Lot.Frontage"] = df_full.groupby("Neighborhood")["Lot.Frontage"].transform(
				    lambda x: x.fillna(x.median()))
				
				# double check if all Nan are dealt
				#print(df_full["Lot.Frontage"].isnull().sum())
				
				
				# Still missing 3 entries, try analyze further:
				#print(df_full[df_full["Lot.Frontage"].isnull()]["Neighborhood"]) # See which neighborhoods these 3 properties are
				
				
				#print(df_full.ix[df_full["Neighborhood"] == "GrnHill", "Lot.Frontage"])
				#print(df_full.ix[df_full["Neighborhood"] == "Landmrk", "Lot.Frontage"])
				# Houses in "GrnHill" and "Landmrk" neighborhood all miss the Lot.Frontage, thereby, fill it with median of all Lot.Frontage.
				
				df_full["Lot.Frontage"] = df_full["Lot.Frontage"].fillna(df_full["Lot.Frontage"].median())
				
				
				# double check if missing values are input correctly
				#print(df_full.ix[df_full["Neighborhood"] == "GrnHill", "Lot.Frontage"])
				#print(df_full.ix[df_full["Neighborhood"] == "Landmrk", "Lot.Frontage"])
				
				# double check if all Nan are dealt
				#print(df_full["Lot.Frontage"].isnull().sum())
				
				
				# Replace missing garage data with 'None' or 0 for houses with no garage
				for col in ('Garage.Type', 'Garage.Finish', 'Garage.Qual', 'Garage.Cond'):
				    df_full[col] = df_full[col].fillna('None')
				#print(df_full[col].isnull().sum())
				    
				for col in ('Garage.Yr.Blt', 'Garage.Area', 'Garage.Cars'):
				    df_full[col] = df_full[col].fillna(0)
				#print(df_full[col].isnull().sum())
				
				# Fix Bsmt missing data
				for col in ("BsmtFin.SF.1", "BsmtFin.SF.2", "Bsmt.Unf.SF", "Total.Bsmt.SF", "Bsmt.Full.Bath", "Bsmt.Half.Bath"):
				    df_full[col] = df_full[col].fillna(0)
				#print(df_full[col].isnull().sum())
				    
				for col in ('Bsmt.Qual', 'Bsmt.Cond', 'Bsmt.Exposure', 'BsmtFin.Type.1', 'BsmtFin.Type.2'):
				    df_full[col] = df_full[col].fillna('None')
				#print(df_full[col].isnull().sum())
				
				
				# Fix Masonry veneer data
				df_full["Mas.Vnr.Type"] = df_full["Mas.Vnr.Type"].fillna("None")
				df_full["Mas.Vnr.Area"] = df_full["Mas.Vnr.Area"].fillna(0)
				df_full['Electrical'] = df_full['Electrical'].fillna(df_full['Electrical'].mode()[0])
				
				
				# Double check if there is still any missing value or Nan.
				
				df_full_na = (df_full.isnull().sum() / len(df_full)) * 100  # calculate the ratio of missing data
				df_full_na = df_full_na.drop(df_full_na[df_full_na == 0].index).sort_values(ascending=False)
				print("Missing", df_full_na.astype(bool).sum(axis=0), "after imputing values.")
				
				
				# ### 2. Transforming numerical variables into categorical variables 
				# Transform a part of the numerical variables to categorical because they merely represent different classes without any ordering. Later I will apply **LabelEncoder for ordinal encoding** or **pd.get_dummies for one-hot encoding** to these variables.
				
				
				# Transform MSSubClass (which identifies the type of dwelling involved in the sale) 
				# and Overall.Cond from numerical to categorical.
				df_full['MS.SubClass'] = df_full['MS.SubClass'].astype(str)
				df_full['Overall.Cond'] = df_full['Overall.Cond'].astype(str)
					
				# #Year and month sold are transformed into categorical features.
				df_full['Yr.Sold'] = df_full['Yr.Sold'].astype(str) # All properties were sold between 2006-2010, so only 4 levels
				df_full['Mo.Sold'] = df_full['Mo.Sold'].astype(str) # All properties were sold between Jan. to Dec.
				
				
				# ### 3. Label Encoding categorical variables with information in their ordering
				# Apply Label Encoder to features associated with quality/condition of living and have ordinal levels
				Ordered_CategoricalFeatures = ('Fireplace.Qu', 'Bsmt.Qual', 'Bsmt.Cond', 'Garage.Qual', 'Garage.Cond', 
				        'Exter.Qual', 'Exter.Cond','Heating.QC', 'Pool.QC', 'Kitchen.Qual', 'BsmtFin.Type.1', 
				        'BsmtFin.Type.2', 'Functional', 'Fence', 'Bsmt.Exposure', 'Garage.Finish', 'Land.Slope',
				        'Lot.Shape', 'Paved.Drive', 'Street', 'Alley', 'Central.Air', 'MS.SubClass', 'Overall.Cond')
				
				# process columns, apply LabelEncoder to categorical features
				for feature in Ordered_CategoricalFeatures:
				    le = LabelEncoder() 
				    le.fit(list(df_full[feature].values)) 
				    df_full[feature] = le.transform(list(df_full[feature].values))
				
				
				#print("There are still", len(df_full.dtypes[df_full.dtypes == "object"].index), "categorical variables.")
				#print(df_full.dtypes[df_full.dtypes == "object"].index)
				
				
				# ### 4. Box Cox / log transformation of skewed features
				numeric_vars = df_full.dtypes[df_full.dtypes != "object"].index
				#print("There are", len(numeric_vars), "numerical variables now.")
				
				# Check the skewness of all continuous variables
				skewed_vars = df_full[numeric_vars].apply(lambda x: skew(x)).sort_values(ascending=False)
				skewness = pd.DataFrame({'Skew' :skewed_vars})
				skewness = skewness.loc[skewness['Skew'].abs() > 0.5] # select features that are more than moderately skewed
				skewed_features = skewness.index
				for feat in skewed_features:
				    lamda = boxcox_normmax(df_full[feat]+1) # the lamda factor
				    df_full[feat] = boxcox1p(df_full[feat], lamda)
				    
				#print("There were", len(skewness), "skewed numerical features Box-Cox transformed.")
				
				
				# ### 5. Normalization of numerical features to avoid ill-conditioning
				# # Scale by Mean/Max-Min or Median/IQR
				df_num = df_full.select_dtypes(include=[np.number])
				# df_norm = (df_num - df_num.mean()) / (df_num.max() - df_num.min())
				# df_full[df_norm.columns] = df_norm
				
				df_norm = robust_scale(df_num, copy=True)
				df_full[df_num.columns] = df_norm
				
				
				# ### 6. Getting dummy variables for categorical features
				#print(df_full.shape)
				df_full = pd.get_dummies(df_full)
			
			
				X_train, X_test, Y_train, Y_test = train_test_split(df_full, Response, train_size=0.7) #random_state=40
				pd.concat([X_test, Y_test], axis=1).to_csv('./data/UI_test.csv', index=False)	
				print("The dimension of full dataset is ", df_full.shape)
				print("The dimension of training set is ", X_train.shape)
				print("The dimension of test set is ", X_test.shape)
				
				# ## Modeling
				
				model = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
				# model = ElasticNetCV(cv=5, random_state=0)
				# model = KernelRidge(alpha=0.9, kernel='polynomial', degree=2, coef0=2.5)
				#model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
				#                                   max_depth=4, max_features='sqrt',
				#                                   min_samples_leaf=15, min_samples_split=10, 
				#                                   loss='huber', random_state =5)
				
				print("Training the Elastic Net model...")
				model.fit(X_train, Y_train)
				#print("R^2 of the model on the test set", model.score(X_test, Y_test))
	
				pickle.dump(model, open('Trained_Model.sav', 'wb'))

				# # Compare the model prediction with actual sale price
				# fig, ax = plt.subplots(figsize=(8, 6))
				# ax.scatter(x=model.predict(X_test), y=Y_test)
				# plt.xlabel("Predicted Sale Price", fontsize=14)
				# plt.ylabel("Actual Sale Price", fontsize=14)
				# lims = [
				#     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
				#     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
				# ]
				
				# # now plot both limits against eachother
				# ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
				# ax.set_aspect('equal')
				# ax.set_xlim(lims)

			else:
				model = pickle.load(open('Trained_Model.sav', 'rb'))
				df_full_test = pd.read_csv(filename)
				Response = df_full_test['SalePrice']
				print("R^2 of the model on the test set from ", filename, "is", model.score(df_full_test.ix[:,0:-1], Response))
				pd.DataFrame(model.predict(df_full_test.ix[:,0:-1])).to_csv('./data/UI_test_prediction.csv', index=False, header=['SalePrice_Prediction'])

				# print(df_full.ix[:,0:-1].shape, Response.shape)

if __name__ == "__main__":
    main()		
