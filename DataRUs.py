from pyod.models.knn import KNN
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from xgboost.sklearn import XGBClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

################################################################################
# PLOTS BASADOS EN TSNE
def plotData(X, y, route):
    reduced = TSNE(n_components=2, n_jobs=-1).fit_transform(X)

    cl0 = np.array([reduced[i] for i in range(len(reduced)) if y[i]=="RESIDENTIAL"])
    cl1 = np.array([reduced[i] for i in range(len(reduced)) if y[i]=="INDUSTRIAL"])
    cl2 = np.array([reduced[i] for i in range(len(reduced)) if y[i]=="PUBLIC"])
    cl3 = np.array([reduced[i] for i in range(len(reduced)) if y[i]=="RETAIL"])
    cl4 = np.array([reduced[i] for i in range(len(reduced)) if y[i]=="OFFICE"])
    cl5 = np.array([reduced[i] for i in range(len(reduced)) if y[i]=="OTHER"])
    cl6 = np.array([reduced[i] for i in range(len(reduced)) if y[i]=="AGRICULTURE"])



    plt.scatter(cl0[:,0], cl0[:,1], color = colors[0], label = "RESIDENTIAL")
    plt.scatter(cl1[:,0], cl1[:,1], color = colors[1], label = "INDUSTRIAL")
    plt.scatter(cl2[:,0], cl2[:,1], color = colors[2], label = "PUBLIC")
    plt.scatter(cl3[:,0], cl2[:,1], color = colors[2], label = "RETAIL")
    plt.scatter(cl4[:,0], cl2[:,1], color = colors[2], label = "OFFICE")
    plt.scatter(cl5[:,0], cl2[:,1], color = colors[2], label = "OTHER")
    plt.scatter(cl2[:,0], cl2[:,1], color = colors[2], label = "AGRICULTURE")

    plt.legend()
    plt.savefig(route+"_2d.png")
    plt.close()

    reduced = TSNE(n_components=3, n_jobs=-1).fit_transform(X)

    d = pd.DataFrame({"x": reduced[:,0], "y": reduced[:,1], "z": reduced[:,2], "labels": y})
    fig = px.scatter_3d(d, x="x", y="y", z="z", color="labels")
    fig.update_traces(marker=dict(size=5,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    plotly.offline.plot(fig, filename=route+"_3d.html", auto_open=True)



####################################################################
# FUNCIÓN PARA LA ELIMINACIÓN DE INSTANCIAS ANÓMALAS
# Está basado el detector de outliers 5-NN para eliminar el 1% instancias
# anómalas
def cleanAnomalies(X,y,perc=0.01):
    detector = KNN(n_neighbors=5, n_jobs=20).fit(X)
    sorted = np.argsort(detector.decision_scores_)[::-1]
    size = len(X)-int(perc*len(X))
    return X[sorted[:size]], y[sorted[:size]]


####################################################################
# LECTURA DE DATOS
data = pd.read_csv('./data/Modelar_UH2020.txt', sep='|')
test = pd.read_csv('./data/Estimar_UH2020.txt', sep='|')


####################################################################
# INFORMACIÓN DEL DATASET
print(data.info())

id_data = data.ID
id_test = test.ID
data2 = data.drop(['ID'], axis=1)
test = test.drop(['ID'], axis=1)

####################################################################
# CORRELACIÓN DE VARIABLES
matriz_corr = data2.corr()
print(matriz_corr)


####################################################################
# DISTRIBUCIÓN POR CLASE

print(data['CLASE'].value_counts())
data['CLASE'].value_counts().plot.bar(title="Distribución por clases")

####################################################################
# VALORES PERDIDOS
data2['CADASTRALQUALITYID'] = data2['CADASTRALQUALITYID'].fillna(data2['CADASTRALQUALITYID'].mode()[0])
data2['MAXBUILDINGFLOOR'] = data2['MAXBUILDINGFLOOR'].fillna(data2['MAXBUILDINGFLOOR'].mode()[0])
test['CADASTRALQUALITYID'] = test['CADASTRALQUALITYID'].fillna(test['CADASTRALQUALITYID'].mode()[0])
test['MAXBUILDINGFLOOR'] = test['MAXBUILDINGFLOOR'].fillna(test['MAXBUILDINGFLOOR'].mode()[0])

###################################################################
# DIVISIÓN EN FEATURES Y TARGET
X = data2.iloc[:,0:54]
y = data2.iloc[:,54:55]
y = np.ravel(y)

# Plot sin tratamiento
plotData(X, y, "raw")

###################################################################
# GENERACIÓN DE DUMMIES EN LA VARIABLE CATEGÓRICA CADASTRALQUALITYID
print("Generando dummies")
dummiesCID = pd.get_dummies(X.CADASTRALQUALITYID)
dummiesCID_test = pd.get_dummies(test.CADASTRALQUALITYID)

###################################################################
# EXTRACCIÓN DE CARACTERÍSTICAS CON PCA SOBRE LAS DUMMIES GENERADAS
print("Aplicando PCA sobre las dummies")
pca = PCA(12)
dummiesCID = pca.fit_transform(dummiesCID)
dummiesCID = pd.DataFrame(dummiesCID)
pca_test = PCA(12)
dummiesCID_test = pca.fit_transform(dummiesCID_test)
dummiesCID_test = pd.DataFrame(dummiesCID_test)


###################################################################
# ESCALADO DE LAS VARIABLES NUMÉRICAS

# Eliminación de CADASTRALQUALITYID para el escalado
X = X.drop(['CADASTRALQUALITYID'], axis=1)
test = test.drop(['CADASTRALQUALITYID'], axis=1)
print("Escalando variables numéricas")
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
scaler_test = preprocessing.StandardScaler()
test = scaler.fit_transform(test)
X = pd.DataFrame(X)
test = pd.DataFrame(test)
# Unión de variables numéricas y las 12 generadas con PCA
X = pd.concat([X,dummiesCID],axis=1)
X = np.array(X)
test = pd.concat([test,dummiesCID_test], axis=1)
test = np.array(test)

# Plot PCA+SCALED
plotData(X, y, "pca-scaled")

###################################################################
# AJUSTE DEL DESBALANCEO: OVERSAMPLING CON SMOTE

# Hago que todas las variables minoritarias tengan 75000 instancias
print("Aplicando SMOTE")
X,y = SMOTE(sampling_strategy = {"INDUSTRIAL": 75000, "PUBLIC": 75000,"RETAIL":75000,"OFFICE":75000,"OTHER":75000, "AGRICULTURE":75000}, random_state=123456789, n_jobs=20, k_neighbors=5).fit_resample(X,y)

# Plot SMOTE
plotData(X, y, "SMOTE")

##################################################################
# LIMPIEZA DE ANOMALÍAS Y RUIDO

print("Eliminando instancias anómalas")
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

# Plot anomalías KNN
plotData(X, y, "anomalias_knn")
print("Instancias por clase:")
print(np.unique(y,return_counts=True))

##################################################################
# MODELO XGB TRAS OPTIMIZAR LOS HIPERPARÁMETROS

print("Entrenando modelo XGB")
xgb = XGBClassifier(
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

model = xgb.fit(X,y)
#pred_train = model.predict(X)
#print(accuracy_score(pred_train,y)
pred_test = model.predict(test)
resultado = pd.DataFrame({'ID':id_test,'CLASE':pred_test})
resultado.to_csv('UGR_Data R Us.txt',sep='|')
