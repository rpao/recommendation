import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer

class recomendacao_usuario:
    def __init__(self, dataset):
        self.metadata = dataset
        
        ## lista de usuarios/venues
        self.users = self.metadata['user_ID'].unique()
        self.venues = self.metadata['venue_ID'].unique()

        self.venue_user = self.metadata.groupby(['venue_ID','user_ID']).size().reset_index().rename(columns={0:'frequence'})
        self.venue_user['peso'] = self.venue_user['frequence']/100

        self.matriz_similaridade()
        
    def gets(self):
        return self.user_similarity, self.venue_similarity

    def matriz_similaridade(self):
        ## criando matrizes esparcas user x venues
        row = []
        col = []
        val = []
        for line in self.metadata.itertuples():
            userIndex = np.where(self.users == line[1])[0][0]
            venueIndex = np.where(self.venues == line[2])[0][0]
            peso = self.venue_user['peso'].get( self.venue_user[ (self.venue_user['venue_ID']  == line[2]) & (self.venue_user['user_ID'] == line[1])].index.tolist()[0])
            row.append(userIndex)
            col.append(venueIndex)
            val.append(peso)

        self.matriz = csr_matrix((val, (row, col)))

        ## medindo similaridade user x user
        ## pairwise utilizando cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)
        self.user_similarity = pairwise_distances(self.matriz, metric='cosine')
        self.venue_similarity = pairwise_distances(self.matriz.T, metric='cosine')

    def predicao_usuario(self, usuario_index):
        semelhantes = []
         
        usuario = self.user_similarity[usuario_index]
 
        distancia = usuario.max()*0.7
 
        for i in range(len(usuario)):
            if i != usuario_index and usuario[i] < distancia:
                semelhantes.append((self.users[i],round(usuario[i],4)))
                
        semelhantes.sort(key=lambda tup: tup[1])
        indices = []
        for u,_ in semelhantes:
            indices += self.metadata.index[self.metadata['user_ID']==u].tolist()

        return self.metadata.ix[indices].drop_duplicates(['user_ID','venue_ID','tips'])
