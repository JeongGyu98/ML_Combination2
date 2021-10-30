# Import the libraries
from operator import length_hint
import random
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn import metrics

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from pyclustering.cluster.clarans import clarans
from sklearn.cluster import estimate_bandwidth
from pyclustering.utils import timedcall;
from pyclustering.cluster import cluster_visualizer_multidim
from sklearn import datasets
from sklearn.metrics import *

from sklearn.metrics import silhouette_score, silhouette_samples

# Two functions to plot the elbow curve
def elbow_curve(model, distortions):
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, 13), distortions)
    plt.grid(True)
    plt.title(model + ' Elbow curve')
    plt.show()

def DBSCAN_elbow_curve(model, distortions, x):
    fig = plt.figure(figsize=(15, 5))
    plt.plot(x, distortions)
    plt.grid(True)
    plt.title(model + ' Elbow curve')
    plt.show()

# The function that performs each models with the given dataset
def best_combination_model(df, scalers=None, encoders=None, models=None):
    # Set the columns to be scaled / encoded
    scale_col = df.columns.tolist()
    encode_col = ''

    # Set the Encoders, Scalers and Models
    if encoders == None:
        encode = [OrdinalEncoder(), OneHotEncoder(), LabelEncoder()]
    else:
        encode = encoders

    if scalers == None:
        scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
        
    else:
        scale = scalers

    if models == None:
        km = KMeans()
        em = GaussianMixture()
        cla = 'clarans'
        db = DBSCAN()
        sc = SpectralClustering()

        classifier = [km, em, cla, db, sc]

    else:
        classifier = models

    # Declare the models' each value

    hyperparams = {
        'k': range(2, 13),
        'eps': [0.01, 0.03,0.05,0.1],
        'min_sample' : [5,10]
    }
     
    # Performing the process
    for i in scale:
        for j in encode:
            # 1. Scaling
            scaler = i
            scaler = pd.DataFrame(scaler.fit_transform(df[scale_col]))
            new_df = scaler

            # 2. Encoding (But, if there's no columns to encode, skip this step)
            if encode_col != '':
                if j == OrdinalEncoder():
                    enc = j
                    enc = enc.fit_transform(df[encode_col])
                    new_df = pd.concat([scaler, enc], axis=1)
                elif j == LabelEncoder():
                    enc = j
                    enc = enc.fit_transform(df[encode_col])
                    new_df = pd.concat([scaler, enc], axis=1)
                else:
                    dum = pd.DataFrame(pd.get_dummies(df[encode_col]))
                    new_df = pd.concat([scaler, dum], axis=1)

            # 3. Modeling
            for model in classifier:

                # For each model, perform the combinations
                # And then, get the best score and best scaler / encoder

                if model == km:
                    
                    distortions = []
                    for k in hyperparams['k']:
                        kmeans = KMeans(n_clusters = k, max_iter=50)
                        cluster = kmeans.fit(new_df)
                        cluster_id = pd.DataFrame(cluster.labels_)
                        distortions.append(kmeans.inertia_)

                        d1 = pd.concat([new_df, cluster_id], axis=1)
                        d1.columns = [0, 1, "cluster"]

                        sns.scatterplot(d1[0], d1[1], hue=d1['cluster'], legend="full")
                        sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], label='Centroids')
                        plt.title("KMeans Clustering with k = " + str(k))
                        plt.legend()
                        plt.show()

                        print('Silhouette score with k = '  + str(k)+ ' : {:.4f}' .format(metrics.silhouette_score(d1.iloc[:,:-1], d1['cluster'])))

                    elbow_curve('Kmeans',distortions)
                    
                elif model == db:

                    eps = hyperparams['eps']
                    sample = hyperparams['min_sample']

                    for i in eps:

                        db = DBSCAN(eps=i, min_samples=4)
                        cluster = db.fit(new_df)
                        cluster_id = pd.DataFrame(cluster.labels_)

                        d2 = pd.DataFrame()
                        d2 = pd.concat([new_df, cluster_id], axis=1)
                        d2.columns = [0, 1, "cluster"]

                        sns.scatterplot(d2[0], d2[1], hue=d2['cluster'], legend="full")
                        plt.title('DBSCAN with eps {}'.format(i))
                        plt.show()

                        print('DBSCAN_Silhouette Score: {:.4f}'.format(metrics.silhouette_score(d2.iloc[:, :-1], d2['cluster'])))
                
                                

                elif model == em:
                    em_silhouette_avg = []

                    for k in hyperparams['k']:
                        model = GaussianMixture(n_components = k)
                        model.fit(new_df)
                        labels = pd.DataFrame(model.fit_predict(new_df))

                    
                        d3 = pd.DataFrame()
                        d3 = pd.concat([new_df, labels], axis = 1)
                        d3.columns = [new_df.columns[0], new_df.columns[1], 'cluster']

                        sns.scatterplot(d3[0], d3[1], hue = d3['cluster'], legend="full")
                        plt.title('EM Clustering with k = {}'.format(k))
                        plt. show()

                        #em_silhouette_avg.append(silhouette_score(new_df, labels, metric='euclidean'))
                        em_silhouette_avg.append(silhouette_score(new_df, labels, metric='manhattan'))

                    elbow_curve('EM(GMM)', em_silhouette_avg)
                
                elif model == cla:
                    data = df.values.tolist()
                    cla_silhouette_avg = []

                    for k in hyperparams['k']:
                        clarans_obj = clarans(random.sample(data, 1000), k, 3, 5)
                        (tks, res) = timedcall(clarans_obj.process)
                        clst = clarans_obj.get_clusters()
                        med = clarans_obj.get_medoids()

                        vis = cluster_visualizer_multidim()
                        vis.append_clusters(clst, new_df, marker="*", markersize=5)

                        labels_c = clarans_obj.get_clusters()
                        labels=pd.DataFrame(labels_c).T.melt(var_name='clusters').dropna()
                        labels['value'] = labels.value.astype(int)
                        labels=labels.sort_values(['value']).set_index('value').values.flatten()

                        cluster_id = pd.DataFrame(labels)
                        d4 = pd.DataFrame()
                        d4 = pd.concat([new_df, cluster_id], axis=1)
                        d4.columns = [0, 1, "cluster"]

                        sns.scatterplot(d4[0], d4[1], hue=d4['cluster'], legend="full")

                        plt.title("Clarans with k = " + str(k))
                        plt.legend()
                        plt.show()

                        print('Silhouette Score(euclidean):', metrics.silhouette_score(new_df.values.tolist(), labels, metric='euclidean'))  

                        cla_silhouette_avg.append(silhouette_score(new_df, labels))       
                    
                    elbow_curve('Clarans', cla_silhouette_avg)

                elif model == sc:
                    silhouette_avg = []

                    for k in hyperparams['k']:
                        model = SpectralClustering(n_clusters=k)
                        model.fit(new_df)
                        cluster_labels = model.labels_

                        cluster_id = pd.DataFrame(cluster_labels)

                        d5 = pd.DataFrame()
                        d5 = pd.concat([new_df, cluster_id], axis=1)
                        d5.columns = [0, 1, "cluster"]
                        sns.scatterplot(d5[0], d5[1], hue=d5['cluster'], legend="full")

                        plt.title("Spectral Clustering with k = " + str(k))
                        plt.legend()
                        plt.show()
		
		        print('Spectral Clustering Silhouette Score with k = k: {}'.format(metrics.silhouette_score(d2.iloc[:, :-1], d2['cluster'])))

                # silhouette score
                silhouette_avg.append(silhouette_score(new_df, cluster_labels))

                elbow_curve(silhouette_avg)

            print("current combination: " + str(i) + ", " + str(j) )
            print("Selected features: ", df.columns.tolist())
            
                        

# importing data
df = pd.read_csv("/Users/kim-jeonggyu/Python/DSProject/housing.csv")
# print(df.head())
df_original = df.copy()
print(df.isnull().sum())

# preprocessing
df.drop(columns=["median_house_value"], inplace=True)
df.total_bedrooms.fillna(df.total_bedrooms.median(), inplace=True)

feature_case = [
            ['longitude', 'households'],
            ['longitude', 'latitude'],
            ['population', 'median_income'],
            ['housing_median_age', 'total_rooms'],
            ['total_rooms', 'total_bedrooms']
        ]
for i in feature_case:
    best_combination_model(df[i])
