import numpy as np
import pandas as pd

from recomendacao_usuario import recomendacao_usuario

from sklearn.model_selection import train_test_split

## base de dados: https://www.kaggle.com/danofer/foursquare-nyc-rest/data

checkins = pd.read_csv('dados/checkins.csv', low_memory = False)
tags = pd.read_csv('dados/corrigido_tags.csv', low_memory = False)
tips = pd.read_csv('dados/corrigido_tips.csv', low_memory = False)

metadata = checkins.merge(tips, on=['venue_ID','user_ID'], how='outer')
##y = range(len(metadata))
##
##train_data, teste_data, y_train, y_test = train_test_split(metadata, y, test_size=0.25, random_state=42)

## quantidade de usuarios/venues/venuesCategory unicos
users = metadata['user_ID'].unique()
venues = metadata['venue_ID'].unique()

n_users = users.shape[0]
n_venues = venues.shape[0]

print 'Numero de usuarios | Numero de Venues\n\t{}\t   |\t {}'.format(n_users,n_venues)
rc = recomendacao_usuario(metadata)
predicoes = rc.predicao_usuario(0)

print predicoes
