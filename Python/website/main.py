# Importations des Bibliotheques
### Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output

### Plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

### Global
import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.decomposition import PCA

### Cluster Algorithm
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

### Mesure Performance
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi


##########################################
##########################################
##########################################
### Machine Learning Code

#---------------------------------------------------------------
#----------------------- VERSION FINAL -------------------------
#---------------------------------------------------------------
# Variable Global
add_button = 0
reset_button  = 0
scatter_plot_ec = None
label_pred_ec = []

### Chargement du dataset
datasetName = None
dataset = None
X = None

### Ensemble Clustering
member_generation = []
algorithms = {'kmeans':0, 'kmeans++':0, 'cah_ward':0, 'cah_complete':0, 'cah_average':0, 'cah_single':0, 'gmm_full':0, 'gmm_tied':0, 'gmm_diag':0, 'gmm_spherical': 0}

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------


##########################################
##########################################
##########################################

# Global
def displayScatterPlot():
    global dataset
    fig = px.scatter(dataset, x='A', y='B', color='target')
    return dcc.Graph(figure=fig)

def displayDataframe():
    global dataset
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataset.columns], style={'fontSize':'23px'})] +

        # Body
        [html.Tr([
            html.Td(dataset.iloc[i][col].round(2)) for col in dataset.columns
        ], style={'fontSize':'20px'}) for i in range(10)]
    , style={'width': '45vw'})

# Clustering Algorithme

def displayKmeans(k=7):
    global dataset
    model = KMeans(n_clusters=k, init="random", max_iter=100, random_state=None, n_init=1).fit(X).labels_
    fig = px.scatter(dataset, x='A', y='B', color=model, title='') 
    return dcc.Graph(figure=fig), model

def displayKmeansPP(k=7):
    global dataset
    model = KMeans(n_clusters=k, init="k-means++", max_iter=100, random_state=None, n_init=1).fit(X).labels_
    fig = px.scatter(dataset, x='A', y='B', color=model, title='') 
    return dcc.Graph(figure=fig), model

def displayCahWard(k=7):
    global dataset
    model = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X).labels_    
    fig = px.scatter(dataset, x='A', y='B', color=model, title='')
    return dcc.Graph(figure=fig), model

def displayCahComplete(k=7):
    global dataset
    model = AgglomerativeClustering(n_clusters=k, linkage='complete').fit(X).labels_    
    fig = px.scatter(dataset, x='A', y='B', color=model, title='')
    return dcc.Graph(figure=fig), model

def displayCahAverage(k=7):
    global dataset
    model = AgglomerativeClustering(n_clusters=k, linkage='average').fit(X).labels_    
    fig = px.scatter(dataset, x='A', y='B', color=model, title='')
    return dcc.Graph(figure=fig), model

def displayCahSingle(k=7):
    global dataset
    model = AgglomerativeClustering(n_clusters=k, linkage='single').fit(X).labels_    
    fig = px.scatter(dataset, x='A', y='B', color=model, title='')
    return dcc.Graph(figure=fig), model

def displayGmmFull(k=7):
    global dataset
    model = GaussianMixture(n_components=k, covariance_type='full').fit(X).predict(X)
    fig = px.scatter(dataset, x='A', y='B', color=model, title='') 
    return dcc.Graph(figure=fig), model

def displayGmmTied(k=7):
    global dataset
    model = GaussianMixture(n_components=k, covariance_type='tied').fit(X).predict(X)
    fig = px.scatter(dataset, x='A', y='B', color=model, title='') 
    return dcc.Graph(figure=fig), model

def displayGmmDiag(k=7):
    global dataset
    model = GaussianMixture(n_components=k, covariance_type='diag').fit(X).predict(X)
    fig = px.scatter(dataset, x='A', y='B', color=model, title='') 
    return dcc.Graph(figure=fig), model

def displayGmmSpherical(k=7):
    global dataset
    model = GaussianMixture(n_components=k, covariance_type='spherical').fit(X).predict(X)
    fig = px.scatter(dataset, x='A', y='B', color=model, title='') 
    return dcc.Graph(figure=fig), model


# Ensemble Clustering
def member_generation_function(X, algorithm='kmeans', nb_partition=5, mix=True):
    '''
    Generating nb_parititions clustering model 
    X : dataset
    nb_partition : number of partition
    mix : Does the result will be merge with an other clustering model in order to use Ensemble Clustering with several algorithm
    '''
    h, _ = X.shape
    partition = []
    global member_generation
    global algorithms
    for i in range(nb_partition):
        k = np.random.randint(2, sqrt(h))
        if algorithm == 'kmeans':
            if mix:
                algorithms['kmeans'] += 1       
            model = KMeans(n_clusters=k, init="random", max_iter=100, random_state=None, n_init=1).fit(X).labels_
        elif algorithm == 'kmeans++':
            if mix:
                algorithms['kmeans++'] += 1   
            model = KMeans(n_clusters=k, init="k-means++", max_iter=100, random_state=None, n_init=1).fit(X).labels_
        elif algorithm == 'cah_ward':
            if mix:
                algorithms['cah_ward'] += 1            
            model = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X).labels_
        elif algorithm == 'cah_complete':
            if mix:
                algorithms['cah_complete'] += 1
            model = AgglomerativeClustering(n_clusters=k, linkage='complete').fit(X).labels_    
        elif algorithm == 'cah_average':
            if mix:
                algorithms['cah_average'] += 1       
            model = AgglomerativeClustering(n_clusters=k, linkage='average').fit(X).labels_    
        elif algorithm == 'cah_single':
            if mix:
                algorithms['cah_single'] += 1
            model = AgglomerativeClustering(n_clusters=k, linkage='single').fit(X).labels_    
        elif algorithm == 'gmm_full':
            if mix:
                algorithms['gmm_full'] += 1  
            model = GaussianMixture(n_components=k, covariance_type='full').fit(X).predict(X)
        elif algorithm == 'gmm_tied':
            if mix:
                algorithms['gmm_tied'] += 1  
            model = GaussianMixture(n_components=k, covariance_type='tied').fit(X).predict(X)
        elif algorithm == 'gmm_diag':
            if mix:
                algorithms['gmm_diag'] += 1   
            model = GaussianMixture(n_components=k, covariance_type='diag').fit(X).predict(X)
        elif algorithm == 'gmm_spherical':
            if mix:
                algorithms['gmm_spherical'] += 1   
            model = GaussianMixture(n_components=k, covariance_type='spherical').fit(X).predict(X)
        partition.append(model)
    if mix:
        member_generation.extend(partition)
    else:
        return partition

def one_hot_encoding(member_generation):
    '''
    Combine multiple partitions in order to form one
    for each partition we represent the membership of individuals in a cluster by a one hot encoding
    
    member_generation : matrix with all partition
    '''
    partitions_one_hot=[]
    for partition_idx, partition in enumerate(member_generation.T):
        nb_individu=len(partition)
        nb_cluster=len(np.unique(partition))
        one_hot_encoding=np.zeros((nb_individu,nb_cluster))
        for individu_idx,individu_affectation_cluster in enumerate(partition):
            one_hot_encoding[individu_idx,individu_affectation_cluster]=1
        partitions_one_hot.append(one_hot_encoding)

    partitions_co_occurence=[partition_one_hot@partition_one_hot.T for partition_one_hot in partitions_one_hot]
    nb_partitions=member_generation.shape[1]
    return (sum(partitions_co_occurence)/nb_partitions).round(2)

def consensus_function(member_generation):
    '''
    Combine multiple partitions in order to form one
    
    member_generation : matrix with all partition
    '''
    h, _ = member_generation.shape
    co_occurence = np.zeros((h, h))
    for i in range(h):
        co_occurence[i, :] = (member_generation == member_generation[i, :]).sum(axis=1)/_
    return co_occurence


def final_clustering_result(co_occurence, n_clusters):
    '''
    Returns the labels, using a clustering algorithm (Spectral Clustering) on the co-association matrix
    
    co_occurence : co-association matrix
    n_clusters : number of cluster to search
    '''
    B = SpectralClustering(n_clusters=n_clusters, random_state=0, affinity="precomputed").fit(co_occurence)  
    return B.labels_

def displayAccuracy(label_real, label_pred):
    '''
    Accuracy of the method using ARI ans NMI
    
    label_real: the real label
    label_pred: the predicted label
    '''
    x = ['ARI', 'NMI']
    y = [round(ari(label_real, label_pred),2), round(nmi(label_real, label_pred),2)]
    fig = go.Figure(data=[go.Bar(x=x, y=y,text=y,textposition='auto',)])
    return dcc.Graph(figure=fig)

def clustering_ensemble(X, algorithm='kmeans', k=7, nb_partition=5, mix=True):
    '''
    Implementation of the Clustering Ensemble method
    
    X : dataset
    algorithm : the name of the algorithm to use 
    nb_partition : number of partition to generate
    mix : if we want to mix several clustering algorithm
    '''
    global member_generation
    global algorithms
    # Clustering Ensemble Algorithm
    if mix:
        member_generation_function(X, algorithm, nb_partition, mix)
        consensus = consensus_function(np.transpose(member_generation))
    else:
        member_generation_simple = member_generation_function(X, algorithm, nb_partition, mix)
        consensus = consensus_function(np.transpose(member_generation_simple))
        
    label_pred = final_clustering_result(consensus, k) 
    
    # Plot the result
    fig = px.scatter(dataset, x="A", y="B", color=label_pred, title='Algorithm : ' + algorithm.upper() + ' - Nb Cluster : ' + str(k) + ' - Nb Partition : ' + str(nb_partition)) 
    return dcc.Graph(figure=fig), label_pred

def displayRatioEC():
    global algorithms
    x = ['K-Means', 'K-Means++', 'CAH-Ward', 'CAH-Complete', 'CAH-Average', 'CAH-Single', 'GMM-Full', 'GMM-Tied', 'GMM-Diag', 'GMM-Spherical']
    y = [algorithms['kmeans'], algorithms['kmeans++'], algorithms['cah_ward'], algorithms['cah_complete'], algorithms['cah_average'], algorithms['cah_single'], algorithms['gmm_full'], algorithms['gmm_tied'], algorithms['gmm_diag'], algorithms['gmm_spherical']]
    fig = go.Figure(data=[go.Bar(x=x, y=y,text=y,textposition='auto',)])
    return dcc.Graph(figure=fig)

##########################################
##########################################
##########################################


external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    # Titre
    html.H1(children='Ensemble Clustering', style={'textAlign': 'center', 'color': '#323256', 'marginBottom' : '5%' }),

    # Row 0
    html.Div(id='row0', children=[
        
        # Loadind Dataset
        html.Div(id='LoadingDataset', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Loading Dataset', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),
                
                html.Div(children=[
                    dcc.Dropdown(
                        id="ListDataset",
                        options=[
                            {'label': 'Aggregation', 'value': '../datasets/Aggregation.txt'},
                            {'label': 'Closer',  'value': '../datasets/closer.txt'},
                            {'label': 'Compact', 'value': '../datasets/compact.txt'},
                            {'label': 'Elongate',  'value': '../datasets/elongate.txt'},
                            {'label': 'Flame',  'value': '../datasets/flame.txt'},
                        ],
                        multi=False,
                        placeholder="Select a Dataset",
                        style={
                            'width' : '100%',
                            'paddingRight' : '7px',
                            'paddingLeft' : '7px',
                            'cursor': 'pointer'
                        }
                    ),
                    dcc.Upload(
                        id="dragDropDataset", 
                        children=([
                            'Drag and Drop or Select a File',
                            html.A('')
                            ]), 
                        style={
                            'marginTop': '15px',
                            'width': '98%',
                            'height': '34px',
                            'border': '1px solid #ccc',
                            'borderRadius': '5px',
                            'display': 'flex',
                            'justifyContent' : 'center',
                            'alignItems' : 'center',
                            'color':'#aaacaf',
                            'marginLeft': '1.2%',
                            'cursor': 'pointer'
                        }
                    ),

                    html.Hr(style={'width': '45vw'}),
                    html.Div(id='contentOutputDataframe', style={
                        'color': '#323256', 
                        'display':'flex', 
                        'justifyContent': 'center',
                        'textAlign': 'center'
                        }),
                ], style={'display' : 'flex', 'flexDirection': 'column'}),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
            # 'alignItems' : 'center'
        }),

        # Scatter Plot
        html.Div(id='ScatterPlot', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Scatter Plot', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),
                html.Div(id='displayScatterPlot'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
        }),
    ], style={'width': '100%', 'display' : 'flex', 'justifyContent' : 'space-between'}),

    # Row 1
    html.Div(id='row1', children=[
        # KMEANS
        html.Div(id='Kmeans', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='K-Means', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),

                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='kmeans_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),
                html.Div(id='kmeans_output'),
                html.Div(id='kmeans_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
            # 'alignItems' : 'center'
        }),

        # KMEANS EC
        html.Div(id='Kmeans_ec', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Ensemble Clustering K-Means', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),
                
                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='kmeans_ec_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),

                # Nombre de partition
                html.H5(children='Number of Partitions', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='kmeans_ec_partitions', 
                    min=5, 
                    max=100, 
                    value=5, 
                    step=5, 
                    marks={5:'5', 10:'10',15:'15',20:'20',25:'25',30:'30',35:'35',40:'40',45:'45',50:'50',55:'55',60:'60',65:'65',70:'70',75:'75',80:'80',85:'85',90:'90',95:'95',100:'100'}
                ),
                
                html.Div(id='kmeans_ec_output'),
                html.Div(id='kmeans_ec_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
        }),
    ], style={'width': '100%', 'marginTop' : '2%', 'display' : 'flex', 'justifyContent' : 'space-between'}),

    # Row 2
    html.Div(id='row2', children=[
        # KMEANS ++
        html.Div(id='Kmeans++', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='K-Means ++', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),

                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='kmeans++_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),
                html.Div(id='kmeans++_output'),
                html.Div(id='kmeans++_performance'),

            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
            # 'alignItems' : 'center'
        }),

        # KMEANS ++ EC
        html.Div(id='Kmeans++_ec', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Ensemble Clustering K-Means ++', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),
                
                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='kmeans++_ec_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),

                # Nombre de partition
                html.H5(children='Number of Partitions', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='kmeans++_ec_partitions', 
                min=5, 
                max=100, 
                value=5, 
                step=5, 
                marks={5:'5', 10:'10',15:'15',20:'20',25:'25',30:'30',35:'35',40:'40',45:'45',50:'50',55:'55',60:'60',65:'65',70:'70',75:'75',80:'80',85:'85',90:'90',95:'95',100:'100'}
                ),
                
                html.Div(id='kmeans++_ec_output'),
                html.Div(id='kmeans++_ec_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
        }),
    ], style={'width': '100%', 'marginTop' : '2%', 'display' : 'flex', 'justifyContent' : 'space-between'}),

    # Row 3
    html.Div(id='row3', children=[
        # CAH - Ward Linkage
        html.Div(id='cah_ward', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='CAH - Ward Linkage', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),

                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='cah_ward_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),
                html.Div(id='cah_ward_output'),
                html.Div(id='cah_ward_performance'),

            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
            # 'alignItems' : 'center'
        }),

        # CAH - Ward Linkage EC
        html.Div(id='cah_ward_ec', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Ensemble Clustering CAH - Ward Linkage', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),
                
                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='cah_ward_ec_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),

                # Nombre de partition
                html.H5(children='Number of Partitions', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='cah_ward_ec_partitions', 
                min=5, 
                max=100, 
                value=5, 
                step=5, 
                marks={5:'5', 10:'10',15:'15',20:'20',25:'25',30:'30',35:'35',40:'40',45:'45',50:'50',55:'55',60:'60',65:'65',70:'70',75:'75',80:'80',85:'85',90:'90',95:'95',100:'100'}
                ),
                
                html.Div(id='cah_ward_ec_output'),
                html.Div(id='cah_ward_ec_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
        }),
    ], style={'width': '100%', 'marginTop' : '2%', 'display' : 'flex', 'justifyContent' : 'space-between'}),

    # Row 4
    html.Div(id='row4', children=[
        # CAH - Complete Linkage
        html.Div(id='cah_complete', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='CAH - Complete Linkage', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),

                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='cah_complete_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),
                html.Div(id='cah_complete_output'),
                html.Div(id='cah_complete_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
            # 'alignItems' : 'center'
        }),

        # CAH - Complete Linkage EC
        html.Div(id='cah_complete_ec', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Ensemble Clustering CAH - Complete Linkage', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),
                
                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='cah_complete_ec_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),

                # Nombre de partition
                html.H5(children='Number of Partitions', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='cah_complete_ec_partitions', 
                min=5, 
                max=100, 
                value=5, 
                step=5, 
                marks={5:'5', 10:'10',15:'15',20:'20',25:'25',30:'30',35:'35',40:'40',45:'45',50:'50',55:'55',60:'60',65:'65',70:'70',75:'75',80:'80',85:'85',90:'90',95:'95',100:'100'}
                ),
                
                html.Div(id='cah_complete_ec_output'),
                html.Div(id='cah_complete_ec_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
        }),
    ], style={'width': '100%', 'marginTop' : '2%', 'display' : 'flex', 'justifyContent' : 'space-between'}),


    # Row 5
    html.Div(id='row5', children=[
        # CAH - Average Linkage
        html.Div(id='cah_average', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='CAH - Average Linkage', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),

                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='cah_average_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),
                html.Div(id='cah_average_output'),
                html.Div(id='cah_average_performance'),

            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
            # 'alignItems' : 'center'
        }),

        # CAH - Average Linkage EC
        html.Div(id='cah_average_ec', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Ensemble Clustering CAH - Average Linkage', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),
                
                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='cah_average_ec_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),

                # Nombre de partition
                html.H5(children='Number of Partitions', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='cah_average_ec_partitions', 
                min=5, 
                max=100, 
                value=5, 
                step=5, 
                marks={5:'5', 10:'10',15:'15',20:'20',25:'25',30:'30',35:'35',40:'40',45:'45',50:'50',55:'55',60:'60',65:'65',70:'70',75:'75',80:'80',85:'85',90:'90',95:'95',100:'100'}
                ),
                
                html.Div(id='cah_average_ec_output'),
                html.Div(id='cah_average_ec_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
        }),
    ], style={'width': '100%', 'marginTop' : '2%', 'display' : 'flex', 'justifyContent' : 'space-between'}),


    # Row 6
    html.Div(id='row6', children=[
        # CAH - Single Linkage
        html.Div(id='cah_single', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='CAH - Single Linkage', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),

                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='cah_single_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),
                html.Div(id='cah_single_output'),
                html.Div(id='cah_single_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
            # 'alignItems' : 'center'
        }),

        # CAH - Single Ward EC
        html.Div(id='cah_single_ec', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Ensemble Clustering CAH - Single Linkage', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),
                
                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='cah_single_ec_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),

                # Nombre de partition
                html.H5(children='Number of Partitions', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='cah_single_ec_partitions', 
                min=5, 
                max=100, 
                value=5, 
                step=5, 
                marks={5:'5', 10:'10',15:'15',20:'20',25:'25',30:'30',35:'35',40:'40',45:'45',50:'50',55:'55',60:'60',65:'65',70:'70',75:'75',80:'80',85:'85',90:'90',95:'95',100:'100'}
                ),
                
                html.Div(id='cah_single_ec_output'),
                html.Div(id='cah_single_ec_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
        }),
    ], style={'width': '100%', 'marginTop' : '2%', 'display' : 'flex', 'justifyContent' : 'space-between'}),

    # Row 7
    html.Div(id='row7', children=[
        # GMM - Full
        html.Div(id='gmm_full', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='GMM - Full', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),

                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='gmm_full_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),
                html.Div(id='gmm_full_output'),
                html.Div(id='gmm_full_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
            # 'alignItems' : 'center'
        }),

        # GMM Full EC
        html.Div(id='gmm_full_ec', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Ensemble Clustering GMM - Full', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),
                
                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='gmm_full_ec_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),

                # Nombre de partition
                html.H5(children='Number of Partitions', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='gmm_full_ec_partitions', 
                min=5, 
                max=100, 
                value=5, 
                step=5, 
                marks={5:'5', 10:'10',15:'15',20:'20',25:'25',30:'30',35:'35',40:'40',45:'45',50:'50',55:'55',60:'60',65:'65',70:'70',75:'75',80:'80',85:'85',90:'90',95:'95',100:'100'}
                ),
                
                html.Div(id='gmm_full_ec_output'),
                html.Div(id='gmm_full_ec_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
        }),
    ], style={'width': '100%', 'marginTop' : '2%', 'display' : 'flex', 'justifyContent' : 'space-between'}),


    # Row 8
    html.Div(id='row8', children=[
        # GMM Tied
        html.Div(id='gmm_tied', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='GMM - Tied', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),

                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='gmm_tied_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),
                html.Div(id='gmm_tied_output'),
                html.Div(id='gmm_tied_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
            # 'alignItems' : 'center'
        }),

        # GMM Tied EC
        html.Div(id='gmm_tied_ec', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Ensemble Clustering GMM - Tied', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),
                
                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='gmm_tied_ec_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),

                # Nombre de partition
                html.H5(children='Number of Partitions', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='gmm_tied_ec_partitions', 
                min=5, 
                max=100, 
                value=5, 
                step=5, 
                marks={5:'5', 10:'10',15:'15',20:'20',25:'25',30:'30',35:'35',40:'40',45:'45',50:'50',55:'55',60:'60',65:'65',70:'70',75:'75',80:'80',85:'85',90:'90',95:'95',100:'100'}
                ),
                
                html.Div(id='gmm_tied_ec_output'),
                html.Div(id='gmm_tied_ec_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
        }),
    ], style={'width': '100%', 'marginTop' : '2%', 'display' : 'flex', 'justifyContent' : 'space-between'}),


        # Row 9
    html.Div(id='row9', children=[
        # GMM diag
        html.Div(id='gmm_diag', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='GMM - Diag', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),

                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='gmm_diag_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),
                html.Div(id='gmm_diag_output'),
                html.Div(id='gmm_diag_performance'),

            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
            # 'alignItems' : 'center'
        }),

        # GMM Diag EC
        html.Div(id='gmm_diag_ec', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Ensemble Clustering GMM - Diag', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),
                
                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='gmm_diag_ec_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),

                # Nombre de partition
                html.H5(children='Number of Partitions', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='gmm_diag_ec_partitions', 
                min=5, 
                max=100, 
                value=5, 
                step=5, 
                marks={5:'5', 10:'10',15:'15',20:'20',25:'25',30:'30',35:'35',40:'40',45:'45',50:'50',55:'55',60:'60',65:'65',70:'70',75:'75',80:'80',85:'85',90:'90',95:'95',100:'100'}
                ),
                
                html.Div(id='gmm_diag_ec_output'),
                html.Div(id='gmm_diag_ec_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
        }),
    ], style={'width': '100%', 'marginTop' : '2%', 'display' : 'flex', 'justifyContent' : 'space-between'}),


        # Row 10
    html.Div(id='row10', children=[
        # GMM Spherical
        html.Div(id='gmm_spherical', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='GMM - Spherical', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),

                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='gmm_spherical_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),
                html.Div(id='gmm_spherical_output'),
                html.Div(id='gmm_spherical_performance'),

            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
            # 'alignItems' : 'center'
        }),

        # GMM Spherical_ EC
        html.Div(id='gmm_spherical_ec', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Ensemble Clustering GMM - Spherical', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),
                
                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='gmm_spherical_ec_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),

                # Nombre de partition
                html.H5(children='Number of Partitions', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='gmm_spherical_ec_partitions', 
                min=5, 
                max=100, 
                value=5, 
                step=5, 
                marks={5:'5', 10:'10',15:'15',20:'20',25:'25',30:'30',35:'35',40:'40',45:'45',50:'50',55:'55',60:'60',65:'65',70:'70',75:'75',80:'80',85:'85',90:'90',95:'95',100:'100'}
                ),
                
                html.Div(id='gmm_spherical_ec_output'),
                html.Div(id='gmm_spherical_ec_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
        }),
    ], style={'width': '100%', 'marginTop' : '2%', 'display' : 'flex', 'justifyContent' : 'space-between'}),


    # Row 11
    html.Div(id='row11', children=[

        # MIX Settings
        html.Div(id='mix_ec', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Ensemble Clustering : Mix of Several Clustering Algorithm', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),
                
                # Nombre de cluster
                html.H5(children='Number of Cluster', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='mix_ec_cluster', 
                    min=2, 
                    max=15, 
                    value=2, 
                    step=1, 
                    marks={2:'2', 3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15'}
                ),

                # Nombre de partition
                html.H5(children='Number of Partitions', style={'textAlign': 'center', 'color': '#323256' }),
                dcc.Slider(id='mix_ec_partitions', 
                    min=5, 
                    max=100, 
                    value=5, 
                    step=5, 
                    marks={5:'5', 10:'10',15:'15',20:'20',25:'25',30:'30',35:'35',40:'40',45:'45',50:'50',55:'55',60:'60',65:'65',70:'70',75:'75',80:'80',85:'85',90:'90',95:'95',100:'100'}
                ),
                
                html.Div(children=[

                    # List of Clustering Algorithm
                    dcc.Dropdown(
                        id="ListClusteringAlgorithm",
                        options=[
                            {'label': 'K-Means', 'value': 'kmeans'},
                            {'label': 'K-Means ++',  'value': 'kmeans++'},
                            {'label': 'CAH - Ward Linkage', 'value': 'cah_ward'},
                            {'label': 'CAH - Complete Linkage', 'value': 'cah_complete'},
                            {'label': 'CAH - Average Linkage', 'value': 'cah_average'},
                            {'label': 'CAH - Single Linkage', 'value': 'cah_single'},
                            {'label': 'GMM - Full',  'value': 'gmm_full'},
                            {'label': 'GMM - Tied',  'value': 'gmm_tied'},
                            {'label': 'GMM - Diag',  'value': 'gmm_diag'},
                            {'label': 'GMM - Spherical',  'value': 'gmm_spherical'},
                        ],
                        multi=False,
                        placeholder="Select a Clustering Algorithm",
                        style={
                            'width' : '75%',
                            'paddingRight' : '7px',
                            'paddingLeft' : '7px',
                            'cursor': 'pointer',
                        }
                    ),


                    html.Button('Add', id='add', n_clicks=0, style={'width':'15%', 'marginRight': '25px', 'backgroundColor':'white', 'border':'1px solid #ccc', 'borderRadius':'5px', 'color':'#323256'}),
                    html.Button('Reset', id='reset', n_clicks=0, style={'width':'15%', 'marginRight': '25px', 'backgroundColor':'white', 'border':'1px solid #ccc', 'borderRadius':'5px', 'color':'#323256'}),

                ], style={
                    'display': 'flex',
                    'marginTop': '13px',
                    'justifyContent' : 'space-between'
                }),
                html.Div(id='mix_ec_settings'),

            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
        }),


        # MIX Display Result
        html.Div(id='mix', children=[
            html.Div(style={'marginTop': '15px'}, children=[
                html.H1(children='Ensemble Clustering : Display Result', style={'textAlign': 'center', 'color': '#323256' }),
                html.Hr(style={'width': '45vw'}),

                html.Div(id='mix_ec_output'),
                html.Div(id='mix_ec_performance'),
            ])
        ], style={
            'backgroundColor' : 'white',
            'height' : '615px',
            'width' : '49%', 
            'overflowY' : 'auto',
            'borderRadius' : '5px',
            'display' : 'flex',
            'boxShadow': '3px 3px 3px #dedede',
            'paddingTop' : '15px',
            'justifyContent' : 'space-around',
            'marginRight' : '7px'
            # 'alignItems' : 'center'
        }),

    ], style={'width': '100%', 'marginTop' : '2%', 'display' : 'flex', 'justifyContent' : 'space-between'}),

], style={
    'backgroundColor': '#f2f2f2',
    'width' : '100vw',
    
    'position' : 'absolute',
    'top' : '0',
    'left' : '0',
    'paddingTop' : '4rem',
    'paddingLeft' : '2rem',
    'paddingRight' : '2rem'
    })


############
# Callback #
############

@app.callback(
    [Output('contentOutputDataframe', 'children'), Output('displayScatterPlot', 'children')],
    [Input('ListDataset', 'value'), Input('dragDropDataset', 'filename')])
def getDataframeFromList(listDataset, dragDropDataset):
    global datasetName
    global dataset
    global X

    if listDataset == None and dragDropDataset == None:
        datasetName = None
        return (None,None)

    if listDataset != None and dragDropDataset != None:
        if listDataset == datasetName:
            datasetName = '../datasets/' + dragDropDataset
        else:
            datasetName = listDataset

    if listDataset != None and dragDropDataset == None:
        datasetName = listDataset
    
    if listDataset == None and dragDropDataset != None:
        datasetName = '../datasets/' + dragDropDataset

    if '.csv' in datasetName:
        dataset = pd.read_csv(datasetName, sep=',', header=None)
    elif '.tsv' in datasetName:
        dataset = pd.read_csv(datasetName, sep='\t', header=None)
    elif '.xls' in datasetName:
        dataset = pd.read_excel(datasetName)
    else:
        dataset = pd.read_csv(datasetName, sep='\t', header=None)
    
    dataset.columns=['A', 'B', 'target']
    X = dataset.loc[:, ['A', 'B']]

    # Analyse en Composante Principale
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    X = pd.DataFrame(data=principalComponents, columns=['A', 'B'])
    dataset = pd.concat([X, dataset[['target']]], axis=1)

    return displayDataframe(), displayScatterPlot()

# Clustering Algorithm
@app.callback(
    [Output('kmeans_output', 'children'), Output('kmeans_performance', 'children')],
    [Input('kmeans_cluster', 'value')])
def kmeans_function(value):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot, label_pred = displayKmeans(value)
    return scatter_plot, displayAccuracy(dataset.target, label_pred)
  
@app.callback(
    [Output('kmeans++_output', 'children'), Output('kmeans++_performance', 'children')],
    [Input('kmeans++_cluster', 'value')])
def kmeansPP_function(value):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot, label_pred =  displayKmeansPP(value)
    return scatter_plot, displayAccuracy(dataset.target, label_pred)
      
@app.callback(
    [Output('cah_ward_output', 'children'), Output('cah_ward_performance', 'children')],
    [Input('cah_ward_cluster', 'value')])
def cah_ward_function(value):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot, label_pred =  displayCahWard(value)
    return scatter_plot, displayAccuracy(dataset.target, label_pred)
      
@app.callback(
    [Output('cah_complete_output', 'children'), Output('cah_complete_performance', 'children')],
    [Input('cah_complete_cluster', 'value')])
def cah_complete_function(value):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot, label_pred =  displayCahComplete(value)
    return scatter_plot, displayAccuracy(dataset.target, label_pred)
      
@app.callback(
    [Output('cah_average_output', 'children'), Output('cah_average_performance', 'children')],
    [Input('cah_average_cluster', 'value')])
def cah_average_function(value):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot, label_pred =  displayCahAverage(value)
    return scatter_plot, displayAccuracy(dataset.target, label_pred)
      
@app.callback(
    [Output('cah_single_output', 'children'), Output('cah_single_performance', 'children')],
    [Input('cah_single_cluster', 'value')])
def cah_single_function(value):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot, label_pred =  displayCahSingle(value)
    return scatter_plot, displayAccuracy(dataset.target, label_pred)
      
@app.callback(
    [Output('gmm_full_output', 'children'), Output('gmm_full_performance', 'children')],
    [Input('gmm_full_cluster', 'value')])
def gmm_full_function(value):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot, label_pred =  displayGmmFull(value)
    return scatter_plot, displayAccuracy(dataset.target, label_pred)
      
@app.callback(
    [Output('gmm_tied_output', 'children'), Output('gmm_tied_performance', 'children')],
    [Input('gmm_tied_cluster', 'value')])
def gmm_tied_function(value):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot, label_pred =  displayGmmTied(value)
    return scatter_plot, displayAccuracy(dataset.target, label_pred)
      
@app.callback(
    [Output('gmm_diag_output', 'children'), Output('gmm_diag_performance', 'children')],
    [Input('gmm_diag_cluster', 'value')])
def gmm_diag_function(value):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot, label_pred =  displayGmmDiag(value)
    return scatter_plot, displayAccuracy(dataset.target, label_pred)
      
@app.callback(
    [Output('gmm_spherical_output', 'children'), Output('gmm_spherical_performance', 'children')],
    [Input('gmm_spherical_cluster', 'value')])
def gmm_spherical_function(value):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot, label_pred =  displayGmmSpherical(value)
    return scatter_plot, displayAccuracy(dataset.target, label_pred)
      

# Ensemble Clustering
@app.callback(
    [Output('kmeans_ec_output', 'children'), Output('kmeans_ec_performance', 'children')],
    [Input('kmeans_ec_cluster', 'value'), Input('kmeans_ec_partitions', 'value')])
def kmeansEC_function(nb_cluster, nb_partition):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot , label_pred  = clustering_ensemble(X, 'kmeans', nb_cluster, nb_partition, False)
    
    return scatter_plot, displayAccuracy(dataset.target, label_pred)

@app.callback(
    [Output('kmeans++_ec_output', 'children'), Output('kmeans++_ec_performance', 'children')],
    [Input('kmeans++_ec_cluster', 'value'), Input('kmeans++_ec_partitions', 'value')])
def kmeansPPEC_function(nb_cluster, nb_partition):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot , label_pred  = clustering_ensemble(X, 'kmeans++', nb_cluster, nb_partition, False)
    
    return scatter_plot, displayAccuracy(dataset.target, label_pred)

@app.callback(
    [Output('cah_ward_ec_output', 'children'), Output('cah_ward_ec_performance', 'children')],
    [Input('cah_ward_ec_cluster', 'value'), Input('cah_ward_ec_partitions', 'value')])
def cah_ward_EC_function(nb_cluster, nb_partition):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot , label_pred  = clustering_ensemble(X, 'cah_ward', nb_cluster, nb_partition, False)
    
    return scatter_plot, displayAccuracy(dataset.target, label_pred)

@app.callback(
    [Output('cah_complete_ec_output', 'children'), Output('cah_complete_ec_performance', 'children')],
    [Input('cah_complete_ec_cluster', 'value'), Input('cah_complete_ec_partitions', 'value')])
def cah_complete_EC_function(nb_cluster, nb_partition):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot , label_pred  = clustering_ensemble(X, 'cah_complete', nb_cluster, nb_partition, False)
    
    return scatter_plot, displayAccuracy(dataset.target, label_pred)

@app.callback(
    [Output('cah_average_ec_output', 'children'), Output('cah_average_ec_performance', 'children')],
    [Input('cah_average_ec_cluster', 'value'), Input('cah_average_ec_partitions', 'value')])
def cah_average_EC_function(nb_cluster, nb_partition):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot , label_pred  = clustering_ensemble(X, 'cah_average', nb_cluster, nb_partition, False)
    
    return scatter_plot, displayAccuracy(dataset.target, label_pred)

@app.callback(
    [Output('cah_single_ec_output', 'children'), Output('cah_single_ec_performance', 'children')],
    [Input('cah_single_ec_cluster', 'value'), Input('cah_single_ec_partitions', 'value')])
def cah_single_EC_function(nb_cluster, nb_partition):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot , label_pred  = clustering_ensemble(X, 'cah_single', nb_cluster, nb_partition, False)
    
    return scatter_plot, displayAccuracy(dataset.target, label_pred)

@app.callback(
    [Output('gmm_full_ec_output', 'children'), Output('gmm_full_ec_performance', 'children')],
    [Input('gmm_full_ec_cluster', 'value'), Input('gmm_full_ec_partitions', 'value')])
def gmm_fullEC_function(nb_cluster, nb_partition):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot , label_pred = clustering_ensemble(X, 'gmm_full', nb_cluster, nb_partition, False)
    
    return scatter_plot, displayAccuracy(dataset.target, label_pred)

@app.callback(
    [Output('gmm_tied_ec_output', 'children'), Output('gmm_tied_ec_performance', 'children')],
    [Input('gmm_tied_ec_cluster', 'value'), Input('gmm_tied_ec_partitions', 'value')])
def gmm_tiedEC_function(nb_cluster, nb_partition):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot , label_pred = clustering_ensemble(X, 'gmm_tied', nb_cluster, nb_partition, False)
    
    return scatter_plot, displayAccuracy(dataset.target, label_pred)

@app.callback(
    [Output('gmm_diag_ec_output', 'children'), Output('gmm_diag_ec_performance', 'children')],
    [Input('gmm_diag_ec_cluster', 'value'), Input('gmm_diag_ec_partitions', 'value')])
def gmm_diagEC_function(nb_cluster, nb_partition):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot , label_pred = clustering_ensemble(X, 'gmm_diag', nb_cluster, nb_partition, False)
    
    return scatter_plot, displayAccuracy(dataset.target, label_pred)

@app.callback(
    [Output('gmm_spherical_ec_output', 'children'), Output('gmm_spherical_ec_performance', 'children')],
    [Input('gmm_spherical_ec_cluster', 'value'), Input('gmm_spherical_ec_partitions', 'value')])
def gmm_sphericalEC_function(nb_cluster, nb_partition):
    global datasetName
    global X
    global dataset
    if datasetName == None:
        return (None, None)
    scatter_plot , label_pred = clustering_ensemble(X, 'gmm_spherical', nb_cluster, nb_partition, False)
    
    return scatter_plot, displayAccuracy(dataset.target, label_pred)


@app.callback(
    [Output('mix_ec_output', 'children'), Output('mix_ec_performance', 'children'), Output('mix_ec_settings', 'children')],
    [Input('mix_ec_cluster', 'value'), Input('mix_ec_partitions', 'value'), Input('ListClusteringAlgorithm', 'value'), Input('add', 'n_clicks'), Input('reset', 'n_clicks')])
def mix_function(nb_cluster, nb_partition, algorithm, add, reset):    
    global add_button
    global reset_button
    global datasetName
    global X
    global dataset
    global scatter_plot_ec
    global label_pred_ec
    global member_generation   
    global algorithms 

    if reset > reset_button:
        scatter_plot_ec = None
        label_pred_ec = []
        reset_button = reset
        member_generation = []    
        algorithms = {'kmeans':0, 'kmeans++':0, 'cah_ward':0, 'cah_complete':0, 'cah_average':0, 'cah_single':0, 'gmm_full':0, 'gmm_tied':0, 'gmm_diag':0, 'gmm_spherical': 0}
        return (None, None, None)


    if add <= add_button or algorithm == None:
        if add_button > 1 and len(label_pred_ec) > 0:
            return scatter_plot_ec, displayAccuracy(dataset.target, label_pred_ec), displayRatioEC()
        else:
            return (None, None, None)

    add_button = add 

    if datasetName == None:
        return (None, None, None)

    scatter_plot_ec , label_pred_ec = clustering_ensemble(X,algorithm,nb_cluster, nb_partition, True)
    return scatter_plot_ec, displayAccuracy(dataset.target, label_pred_ec), displayRatioEC()

if __name__ == '__main__':
    app.run_server(debug=True)
