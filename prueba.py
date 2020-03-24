import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from anomaly_cleaning import cleanAnomalies
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

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
test = pd.read_csv('./data/Estimar_UH2020.txt', sep='|')
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



X,y = SMOTE(sampling_strategy = {"INDUSTRIAL": 75000, "PUBLIC": 75000,"RETAIL":75000,"OFFICE":75000,"OTHER":75000, "AGRICULTURE":75000}, random_state=123456789, n_jobs=20, k_neighbors=5).fit_resample(X,y)





#from sklearn.model_selection import train_test_split

#X, X_test, y, y_test = train_test_split(X,y, test_size = 0.3, stratify=y,random_state=77145416)



print("Cleaning anomalies...")
ind_res = np.where(y=="RESIDENTIAL")[0]
ind_ind = np.where(y=="INDUSTRIAL")[0]
ind_pub = np.where(y=="PUBLIC")[0]
ind_ret = np.where(y=="RETAIL")[0]
ind_off = np.where(y=="OFFICE")[0]
ind_ot = np.where(y=="OTHER")[0]
ind_agr = np.where(y=="AGRICULTURE")[0]
X1, y1 = cleanAnomalies(X[ind_res], y[ind_res])
X2, y2 = cleanAnomalies(X[ind_ind], y[ind_ind])
X3, y3 = cleanAnomalies(X[ind_pub], y[ind_pub])
X4, y4 = cleanAnomalies(X[ind_ret], y[ind_ret])
X5, y5 = cleanAnomalies(X[ind_off], y[ind_off])
X6, y6 = cleanAnomalies(X[ind_ot], y[ind_ot])
X7, y7 = cleanAnomalies(X[ind_agr], y[ind_agr])
X = np.concatenate((X1,X2), axis=0)
X = np.concatenate((X,X3), axis=0)
X = np.concatenate((X,X4), axis=0)
X = np.concatenate((X,X5), axis=0)
X = np.concatenate((X,X6), axis=0)
X = np.concatenate((X,X7), axis=0)
y = np.concatenate((y1,y2), axis=0)
y = np.concatenate((y,y3), axis=0)
y = np.concatenate((y,y4), axis=0)
y = np.concatenate((y,y5), axis=0)
y = np.concatenate((y,y6), axis=0)
y = np.concatenate((y,y7), axis=0)

print("Instancias por clase:")
print(np.unique(y,return_counts=True))



import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score


xgb1 = XGBClassifier(
 learning_rate =0.1,
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

#dtrain = xgb.DMatrix(X,label = y)
#dtest = xgb.DMatrix(X_test, )
model = xgb1.fit(X,y)
pred_train = model.predict(X)
print(accuracy_score(pred_train,y))
