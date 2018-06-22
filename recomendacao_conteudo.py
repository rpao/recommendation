import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# pega uma venue_ID e retorna os venues_ID mais similares
def get_recommendations(venues, venue_ID, cosine_sim, indices):
    # pega indices das venues que combinam
    try:
        idx = indices[venue_ID]
    except KeyError:
        return pd.DataFrame()

    # pairwsie similarity scores de todas as venues com a venue recebida
    sim_scores = list(enumerate(cosine_sim[idx]))

    # ordena pelo score
    try:
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    except:
        sim_scores = sorted(sim_scores, key=lambda x: x[1].all(), reverse=True)

    # pega as 10 mais similares
    sim_scores = sim_scores[1:10]

    # pega os indices
    venue_indices = [i[0] for i in sim_scores]

    # retorna top 10 venues similares
    return venues.iloc[venue_indices].drop_duplicates()

## recomendacao baseada em palavra-chave - segundo tips/tags do local
tips = pd.read_csv('dados/corrigido_tips.csv', low_memory = False).drop(['user_ID'], axis = 1)
tags = pd.read_csv('dados/corrigido_tags.csv', low_memory = False)

try:
    #Define um objeto TF-IDF vetorizado. Remove preposicoes (ingles)
    tfidf = TfidfVectorizer(stop_words='english')

    #Substitui NaN com string vazia
    tips['tips'] = tips['tips'].fillna('')
    tags['tags'] = tags['tags'].fillna('')

    #Constroi a matriz TF-IDF (filtra e transforma dados)
    tfidf_matrix_tips = tfidf.fit_transform(tips['tips'])
    tfidf_matrix_tags = tfidf.fit_transform(tags['tags'])

except UnicodeDecodeError as e:
    print "UnicodeDecodeError error: {}".format(e)
    exit()

# matriz de cosine-similarity
tips_cosine_sim = linear_kernel(tfidf_matrix_tips, tfidf_matrix_tips)
tags_cosine_sim = linear_kernel(tfidf_matrix_tags, tfidf_matrix_tags)

#Constroi um map reverso de indices e venue_ID (sem duplicatas)
tips_indices = pd.Series(tips.index, index=tips['venue_ID']).drop_duplicates()
tags_indices = pd.Series(tags.index, index=tags['venue_ID']).drop_duplicates()

venue_id = int(raw_input("Informe uma venue: "))

try:
    all_data = tips.merge(tags, on='venue_ID',how='outer')#, lsuffix = "overlap")
    venue_data = all_data.loc[all_data['venue_ID'] == venue_id]
except:
    print "No venue id found..."
    exit()

tips_recomendados = get_recommendations(tips, venue_id, tips_cosine_sim, tips_indices)
if (tips_recomendados.shape[0] > 0):
    tips_recomendados = tips_recomendados.merge(tags, on='venue_ID',how='left')

tags_recomendados = get_recommendations(tags, venue_id, tags_cosine_sim, tags_indices)
if (tags_recomendados.shape[0] > 0):
    tags_recomendados = tags_recomendados.merge(tips, on='venue_ID',how='left')

venue_data.to_csv("resultados/conteudo/recomendados_"+repr(venue_id)+".csv")
tips_recomendados.to_csv("resultados/conteudo/recomendados_tips_"+repr(venue_id)+".csv")
tags_recomendados.to_csv("resultados/conteudo/recomendados_tags_"+repr(venue_id)+".csv")
