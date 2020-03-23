import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from anomaly_cleaning import cleanAnomalies
from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing
import seaborn as sns
'''
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
'''

data = pd.read_csv('./data/Modelar_UH2020.txt', sep='|')
id_data = data.ID
data2 = data.drop(['ID'], axis=1)

data2['CADASTRALQUALITYID'] = data2['CADASTRALQUALITYID'].fillna(data2['CADASTRALQUALITYID'].mode()[0])
data2['MAXBUILDINGFLOOR'] = data2['MAXBUILDINGFLOOR'].fillna(data2['MAXBUILDINGFLOOR'].mode()[0])

X = data2.iloc[:,0:54]
y = data2.iloc[:,54:55]
y = np.ravel(y)
dummiesCID = pd.get_dummies(X.CADASTRALQUALITYID)
pca = PCA(12)
dummiesCID = pca.fit_transform(dummiesCID)
dummiesCID = pd.DataFrame(dummiesCID)
X = X.drop(['CADASTRALQUALITYID'], axis=1)
#SCALING
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)
X = pd.concat([X,dummiesCID],axis=1)
X = np.array(X)
X,y = SMOTE(sampling_strategy = {"INDUSTRIAL": 60000, "PUBLIC": 60000,"RETAIL":60000,"OFFICE":60000,"OTHER":60000, "AGRICULTURE":60000}, random_state=123456789, n_jobs=20, k_neighbors=5).fit_resample(X,y)





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, stratify=y,random_state=77145416)



print("Cleaning anomalies...")
ind_res = np.where(y_train=="RESIDENTIAL")[0]
ind_ind = np.where(y_train=="INDUSTRIAL")[0]
ind_pub = np.where(y_train=="PUBLIC")[0]
ind_ret = np.where(y_train=="RETAIL")[0]
ind_off = np.where(y_train=="OFFICE")[0]
ind_ot = np.where(y_train=="OTHER")[0]
ind_agr = np.where(y_train=="AGRICULTURE")[0]
X1, y1 = cleanAnomalies(X_train[ind_res], y_train[ind_res])
X2, y2 = cleanAnomalies(X_train[ind_ind], y_train[ind_ind])
X3, y3 = cleanAnomalies(X_train[ind_pub], y_train[ind_pub])
X4, y4 = cleanAnomalies(X_train[ind_ret], y_train[ind_ret])
X5, y5 = cleanAnomalies(X_train[ind_off], y_train[ind_off])
X6, y6 = cleanAnomalies(X_train[ind_ot], y_train[ind_ot])
X7, y7 = cleanAnomalies(X_train[ind_agr], y_train[ind_agr])
X_train = np.concatenate((X1,X2), axis=0)
X_train = np.concatenate((X_train,X3), axis=0)
X_train = np.concatenate((X_train,X4), axis=0)
X_train = np.concatenate((X_train,X5), axis=0)
X_train = np.concatenate((X_train,X6), axis=0)
X_train = np.concatenate((X_train,X7), axis=0)
y_train = np.concatenate((y1,y2), axis=0)
y_train = np.concatenate((y_train,y3), axis=0)
y_train = np.concatenate((y_train,y4), axis=0)
y_train = np.concatenate((y_train,y5), axis=0)
y_train = np.concatenate((y_train,y6), axis=0)
y_train = np.concatenate((y_train,y7), axis=0)

print("Instancias por clase:")
print(np.unique(y_train,return_counts=True))

'''
print("EditedNearestNeighbours...")
X_train, y_train = EditedNearestNeighbours(sampling_strategy="not minority", n_neighbors=15, n_jobs=20, kind_sel="mode").fit_resample(X_train, y_train)
print("Numero de instancias: " + str(len(X_train)))
print("Instancias por clase:")
print(np.unique(y_train,return_counts=True))
'''


import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score

'''
#X_train, y_train = IPF(X_train, y_train)
#print("Numero de instancias: " + str(len(X_train)))
#print("Instancias por clase:")
#print(np.unique(y_train,return_counts=True))
'''
xgb1 = XGBClassifier(
 learning_rate =0.05,
 n_estimators=2500,
 max_depth=10,
 min_child_weight=3,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softprob',
 reg_alpha=0.005,
 nthread=10,
 scale_pos_weight=2,
 seed=27)
#dtrain = xgb.DMatrix(X_train,label = y_train)
#dtest = xgb.DMatrix(X_test, )
model = xgb1.fit(X_train,y_train)
pred = model.predict(X_test)
print(accuracy_score(pred,y_test))
'''
## Tune max_depth and min_child_weight
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softprob', nthread=4, scale_pos_weight=1, seed=27),
 param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train,y_train)
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softprob', nthread=4, scale_pos_weight=1,seed=27),
 param_grid = param_test3, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch3.fit(X_train,y_train)
print(gsearch3.cv_results_)

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softprob', nthread=4, scale_pos_weight=1,seed=27),
 param_grid = param_test4, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch4.fit(X_train,y_train)
print(gsearch4.best_params_, gsearch4.best_score_)

param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch6.fit(X_train,y_train)
print(gsearch6.best_params_, gsearch6.best_score_)
'''
