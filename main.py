import numpy as np
import pandas as pd

from recomendacao_usuario import recomendacao_usuario
from recomendacao_conteudo import recomendacao_conteudo

from sklearn.model_selection import train_test_split

## base de dados: https://www.kaggle.com/danofer/foursquare-nyc-rest/data

def recomendacao_por_usuario():
    checkins = pd.read_csv('dados/checkins.csv', low_memory = False)

    ## quantidade de usuarios/venues/venuesCategory unicos
    users = checkins['user_ID'].unique()
##    venues = checkins['venue_ID'].unique()
##    n_users = users.shape[0]
##    n_venues = venues.shape[0]
##    print 'Numero de usuarios | Numero de Venues\n\t{}\t   |\t {}'.format(n_users,n_venues)
    index_usuario = int(raw_input("Informe um Usuario: "))
    usuario = users[index_usuario]
    i = checkins.index[checkins['user_ID'] == usuario]
    p = checkins.ix[i]
    print "Usuario:\n",p

    ru = recomendacao_usuario(checkins)    
    predicoes = ru.predicao_usuario(index_usuario)
    predicoes.to_csv("resultados/usuario/predicoes_"+repr(usuario)+".csv")
    
    return predicoes

def recomendacao_por_conteudo():
    tags = pd.read_csv('dados/corrigido_tags.csv', low_memory = False)
    tips = pd.read_csv('dados/corrigido_tips.csv', low_memory = False)

    tips = tips.merge(tags, on='venue_ID',how='left')
    
    ## quantidade de usuarios/venues/venuesCategory unicos
    venues = tips['venue_ID'].unique()
    venue_id = venues[int(raw_input("Informe uma venue: "))]
    i = tips.index[tips['venue_ID'] == venue_id]
    p = tips.ix[i]
    print "Venue:\n",p
    
    rc = recomendacao_conteudo(tips, venue_id)
    predicoes = rc.predicao()

    predicoes.to_csv("resultados/conteudo/predicoes_"+repr(venue_id)+".csv")
    return predicoes

opcao = int(raw_input("Tipo de recomendacao: \n[ 1 ] usuario\n[ 2 ] Item\n>>"))

if opcao == 1:
    predicoes = recomendacao_por_usuario()
elif opcao == 2:
    predicoes = recomendacao_por_conteudo()
else:
    predicoes =  "Opcao invalida..."

print predicoes
