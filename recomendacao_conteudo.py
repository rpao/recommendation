import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# pega uma venue_ID e retorna os venues_ID mais similares
class recomendacao_conteudo:
    def __init__(self, tip, venue):
        self.tips = tip.drop(['user_ID'], axis = 1)
        
        self.venue_id = venue
        
        try:
            #Define um objeto TF-IDF vetorizado. Remove preposicoes (ingles)
            self.tfidf = TfidfVectorizer(stop_words='english')

            #Substitui NaN com string vazia
            self.tips['tips'] = self.tips['tips'].fillna('')
            self.tips['tags'] = self.tips['tags'].fillna('')

            #Constroi a matriz TF-IDF (filtra e transforma dados)
            self.tfidf_matrix_tips = self.tfidf.fit_transform(self.tips['tips'])

        except UnicodeDecodeError as e:
            return ("UnicodeDecodeError error: {}".format(e), -1)
    
    def get_recommendations(self, venues, venue_ID, cosine_sim, indices):
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

    def predicao(self):
        # matriz de cosine-similarity
        tips_cosine_sim = linear_kernel(self.tfidf_matrix_tips, self.tfidf_matrix_tips)

        #Constroi um map reverso de indices e venue_ID (sem duplicatas)
        tips_indices = pd.Series(self.tips.index, index=self.tips['venue_ID']).drop_duplicates()

        self.tips_recomendados = self.get_recommendations(self.tips, self.venue_id, tips_cosine_sim, tips_indices)

        return self.tips_recomendados.drop_duplicates(['venue_ID'])
