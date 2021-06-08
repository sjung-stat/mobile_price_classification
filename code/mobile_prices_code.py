
############# Preparation #############
# Import packages
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
train = pd.read_csv('train.csv')
train.head()
train.info()





############# Exploratory Data Analysis #############
# For data visualization,create a copy of the original dataframe and convert data types of 8 variables
# And change the levels of categorical variables
new_train = train.copy()
col_names = ['blue', 'dual_sim', 'four_g', 'n_cores', 'three_g', 'touch_screen', 'wifi', 'price_range']
for col in col_names:
    new_train[col] = new_train[col].astype('category',copy=False)
new_train.blue.replace((1, 0), ('Yes', 'No'), inplace=True)
new_train.dual_sim.replace((1, 0), ('Yes', 'No'), inplace=True)
new_train.four_g.replace((1, 0), ('Yes', 'No'), inplace=True)
new_train.three_g.replace((1, 0), ('Yes', 'No'), inplace=True)
new_train.touch_screen.replace((1, 0), ('Yes', 'No'), inplace=True)
new_train.wifi.replace((1, 0), ('Yes', 'No'), inplace=True)
new_train.price_range.replace((3, 2, 1, 0), ('Very high', 'High', 'Medium', 'Low'), inplace=True)

# Visualize continuous variables
new_train.hist(bins=50, figsize=(20,15))
plt.show(block=False)







fig = plt.subplots(figsize=(10, 10))
sns.heatmap(new_train.corr(), annot = True, cbar = True, square = True, cmap="OrRd", annot_kws = {'size': 8.5})




# Visualize the first four categorical variables
col_names = ['blue', 'dual_sim', 'four_g', 'n_cores', 'three_g', 'touch_screen', 'wifi', 'price_range']
fig, axs = plt.subplots(ncols=4, figsize=(15,5))
j = 0
for i in col_names[:4]:
    sns.countplot(x=i, palette="ch:.25", data=new_train, ax=axs[j])
    axs[j].set_ylabel("")
    axs[j].set_xlabel("")
    axs[j].set_title(i)
    j = j+1
    
# Visualize the other last first four categorical variables
fig, axs = plt.subplots(ncols=4, figsize=(15,5))
j = 0
for i in col_names[4:]:
    sns.countplot(x=i, palette="ch:.25", data=new_train, ax=axs[j])
    axs[j].set_ylabel("")
    axs[j].set_xlabel("")
    axs[j].set_title(i)
    j = j+1






# Scatterplots of continuous variables based on price range 
fig, axs = plt.subplots(ncols=3, figsize=(15,5))
sns.scatterplot(x="pc", y="fc", hue="price_range", data=new_train, ax=axs[0])
axs[0].set_ylabel("Front Camera mega pixels")
axs[0].set_xlabel("Primary Camera mega pixels")
sns.scatterplot(x="px_height", y="px_width", hue="price_range", data=new_train, ax=axs[1])
axs[1].set_ylabel("Pixel resolution width")
axs[1].set_xlabel("Pixel resolution height")
sns.scatterplot(x="sc_h", y="sc_w", hue="price_range", data=new_train, ax=axs[2])
axs[2].set_ylabel("Screen width")
axs[2].set_xlabel("Screen height")





# Two categorical variables provide information on their generation ('three_g', 'four_g').
# Combine the two and create a new variable 
new_train["generation"] = ""
for i in range(new_train.shape[0]):
    if new_train["three_g"][i] == 'Yes' and new_train["four_g"][i] == 'Yes':
        new_train["generation"][i] = '4G'
    elif new_train["three_g"][i] == 'Yes' and new_train["four_g"][i] == 'No':
        new_train["generation"][i] = '3G'
    else:
        new_train["generation"][i] = '2G'
drop_var = ["three_g", "four_g"]
new_train = new_train.drop(drop_var, 1)        
new_train




# Create three plots against price range: front camera mega pixels, primary camera mega pixels, camera mega pixels
fig, axs = plt.subplots(ncols=3, figsize=(15,5))
new_train_copy = new_train.copy()
new_train_copy['cam_pix'] = new_train_copy["fc"] + new_train_copy["pc"]
sns.stripplot(x="price_range", order=["Low", "Medium", "High", "Very high"], y="fc", data=new_train_copy, ax=axs[0])
axs[0].set_ylabel("Front Camera mega pixels")
axs[0].set_xlabel("Price Range")
sns.stripplot(x="price_range", order=["Low", "Medium", "High", "Very high"], y="pc", data=new_train_copy, ax=axs[1])
axs[1].set_ylabel("Primary Camera mega pixels")
axs[1].set_xlabel("Price Range")
sns.stripplot(x="price_range", order=["Low", "Medium", "High", "Very high"], y="cam_pix", data=new_train_copy, ax=axs[2])
axs[2].set_ylabel("Camera mega pixels")
axs[2].set_xlabel("Price Range")




# Create a new variable ('pixel_dimension') by multiplying 'px_width' and 'px_height' and drop them as they are correlated
new_train["pixel_dimension"] = new_train["px_width"] * new_train["px_height"]
drop_var = ["px_width", "px_height"]
new_train = new_train.drop(drop_var, 1)

# Create another new variable ('screen_dimension') by multiplying 'sc_w' and 'sc_h' and drop them as they are correlated
new_train["screen_dimension"] = new_train["sc_w"] * new_train["sc_h"]
drop_var = ["sc_w", "sc_h"]
new_train = new_train.drop(drop_var, 1)



# Create three plots against price range: RAM, pixel dimension, screen dimension
fig, axs = plt.subplots(ncols=3, figsize=(15,5))
sns.stripplot(x="price_range", order=["Low", "Medium", "High", "Very high"], y="ram", data=new_train, ax=axs[0])
axs[0].set_ylabel("RAM")
axs[0].set_xlabel("Price Range")
sns.stripplot(x="price_range", order=["Low", "Medium", "High", "Very high"], y="pixel_dimension", data=new_train, ax=axs[1])
axs[1].set_ylabel("Pixel Dimension")
axs[1].set_xlabel("Price Range")
sns.stripplot(x="price_range", order=["Low", "Medium", "High", "Very high"], y="screen_dimension", data=new_train, ax=axs[2])
axs[2].set_ylabel("Screen Dimension")
axs[2].set_xlabel("Price Range")



# Boxplot of the number against battery power based on price range
sns.boxplot(x="n_cores", y="battery_power", hue="price_range", hue_order=["Low", "Medium", "High", "Very high"], data=new_train, palette="coolwarm")




# Plot stacked barplots for the categorical variables. 
df1 = pd.crosstab(new_train["price_range"], new_train["generation"]) / new_train.shape[0]
df2 = pd.crosstab(new_train["price_range"], new_train["n_cores"]) / new_train.shape[0]
df3 = pd.crosstab(new_train["price_range"], new_train["blue"]) / new_train.shape[0]
df4 = pd.crosstab(new_train["price_range"], new_train["dual_sim"]) / new_train.shape[0]
df5 = pd.crosstab(new_train["price_range"], new_train["touch_screen"]) / new_train.shape[0]
df6 = pd.crosstab(new_train["price_range"], new_train["wifi"]) / new_train.shape[0]
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, ncols=3, figsize=(15,12))
fig.suptitle('Categorical Variables w/ price range')
df1.plot(kind='bar', stacked=True, ax=ax1)
df2.plot(kind='bar', stacked=True, ax=ax2)
df3.plot(kind='bar', stacked=True, ax=ax3)
df4.plot(kind='bar', stacked=True, ax=ax4)
df5.plot(kind='bar', stacked=True, ax=ax5)
df6.plot(kind='bar', stacked=True, ax=ax6)




############# Feature Engineering #############
from mobile_feat_eng import feat_eng
feat_eng(train)

# Drop rows that contain 0 in the new variables as they don't make sense
train = train.loc[~((train['pixel_dimension'] == 0) | (train['screen_dimension'] == 0))]

# Convert the object into integer type
train['generation'] = train['generation'].astype(str).astype(int)
train.info()

# Split data into training and test (validation) sets. 
X = train.drop(["price_range"], axis=1)
Y = train["price_range"].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Standardize the continuous features 
col_names = list(X_train.columns)
features = X_train[col_names]
features_test = X_test[col_names]
ct = ColumnTransformer([
        ('somename', StandardScaler(), ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'pixel_dimension', 'screen_dimension', 'camera_pixels'])
    ], remainder='passthrough')
X_train_transformed = ct.fit_transform(features)
X_test_transformed = ct.transform(features_test)



############# Model Building #############

### Softmax Regression
# Without tuning
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, max_iter=3000)
softmax_reg.fit(X_train_transformed, Y_train)
print(softmax_reg.score(X_train_transformed, Y_train))   # train error
print(softmax_reg.score(X_test_transformed, Y_test))     # test error
predictions = softmax_reg.predict(X_test_transformed)
print(classification_report(Y_test, predictions))

# Grid search for softmax regression
params = {'C':[1, 5, 10, 15, 50, 100], 'tol': [0.001, 0.0001, 0.005]}
log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=3000, penalty='l2')
clf = GridSearchCV(log_reg, params, refit='True', n_jobs=1, verbose=3, cv=5)
clf.fit(X_train_transformed, Y_train)
print(clf.score(X_train_transformed, Y_train))   # train error
print(clf.score(X_test_transformed, Y_test))     # test error
predictions = clf.predict(X_test_transformed)
print(classification_report(Y_test, predictions))
print(clf.best_params_)



### Neural Network w/ Scikit Learn
# Without tuning
NN = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(150, 10), random_state=1, max_iter=3000)
NN.fit(X_train_transformed, Y_train)
print(NN.score(X_train_transformed,Y_train))   # train error
print(NN.score(X_test_transformed,Y_test))     # test error
y_true, y_pred = Y_test, NN.predict(X_test_transformed)
print(classification_report(y_true, y_pred))

# Grid search for neural network
mlp = MLPClassifier(max_iter=100)
parameter_space = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (150, 10), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.05, 0.1, 1],
    'learning_rate': ['constant','adaptive']
}
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train_transformed, Y_train)
print(clf.score(X_train_transformed,Y_train))   # train error
print(clf.score(X_test_transformed,Y_test))     # test error
y_true, y_pred = Y_test, clf.predict(X_test_transformed)
print(classification_report(y_true, y_pred))
print(clf.best_params_)


### Support Vector Machine
# Without tuning
svm=SVC(random_state=1, coef0=0, C=1, gamma= 'scale', kernel='rbf', decision_function_shape='ovo')
svm.fit(X_train_transformed, Y_train)
print(svm.score(X_train_transformed, Y_train))   # train error
print(svm.score(X_test_transformed, Y_test))     # test error
predictions = svm.predict(X_test_transformed)
print(classification_report(Y_test, predictions))

# Grid search for support vector machine
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.0001], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
clf_svc = GridSearchCV(SVC(), tuned_parameters, refit = True, verbose = 3)
clf_svc.fit(X_train_transformed, Y_train)
print(clf.best_params_)
means = clf_svc.cv_results_['mean_test_score']
stds = clf_svc.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf_svc.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print(clf_svc.score(X_train_transformed, Y_train))   # train error
print(clf_svc.score(X_test_transformed, Y_test))     # test error
y_true, y_pred = Y_test, clf_svc.predict(X_test_transformed)
print(classification_report(y_true, y_pred))
print(clf_svc.best_params_)



### Random Forest
# Without tuning
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=20, n_jobs=-1, oob_score=True)
rnd_clf.fit(X_train_transformed, Y_train)
print(rnd_clf.score(X_train_transformed, Y_train))   # train error
print(rnd_clf.score(X_test_transformed, Y_test))     # test error
y_true, y_pred = Y_test, rnd_clf.predict(X_test_transformed)
print(classification_report(y_true, y_pred))


### Gaussian Naive Bayes
gnb = GaussianNB(priors=None, var_smoothing=1e-09)
gnb.fit(X_train_transformed, Y_train)
print(gnb.score(X_train_transformed,Y_train))   # train error
print(gnb.score(X_test_transformed,Y_test))     # test error
y_true, y_pred = Y_test, gnb.predict(X_test_transformed)
print(classification_report(y_true, y_pred))


### KNN
knn = KNeighborsClassifier(n_neighbors=9, weights='uniform', algorithm='auto')
knn.fit(X_train_transformed,Y_train)
print(knn.score(X_train_transformed,Y_train))   # train error
print(knn.score(X_test_transformed,Y_test))     # test error
y_true, y_pred = Y_test, knn.predict(X_test_transformed)
print(classification_report(y_true, y_pred))





############# Discussion #############
# PCA plot
pca = PCA().fit(X_train_transformed)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# Softmax regression: Top 10 variables that affect the target variable the most
pd.Series(abs(softmax_reg.coef_[0]), index=features.columns).nlargest(10).plot(kind='barh')

# Support vector machine: Top 10 variables that affect the target variable the most
svm=SVC(random_state=1, C=10, kernel='linear')
svm.fit(X_train_transformed, Y_train)
pd.Series(abs(svm.coef_[0]), index=features.columns).nlargest(10).plot(kind='barh')