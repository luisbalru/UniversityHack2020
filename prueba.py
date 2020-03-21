import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
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
pca = PCA(n_components=12)
dummiesCID = pca.fit_transform(dummiesCID)
dummiesCID = pd.DataFrame(dummiesCID)
X = X.drop(['CADASTRALQUALITYID'], axis=1)
X = pd.concat([X,dummiesCID],axis=1)
X = np.array(X)
X,y = SMOTE(sampling_strategy = {"INDUSTRIAL": 60000, "PUBLIC": 60000,"RETAIL":60000,"OFFICE":60000,"OTHER":60000, "AGRICULTURE":60000}, random_state=123456789, n_jobs=20, k_neighbors=5).fit_resample(X,y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, stratify=y,random_state=77145416)

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softprob',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
#dtrain = xgb.DMatrix(X_train,label = y_train)
#dtest = xgb.DMatrix(X_test, )
model = xgb1.fit(X_train,y_train)
pred = model.predict(X_test)
print(accuracy_score(pred,y_test))
