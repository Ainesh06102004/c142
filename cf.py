import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("final.csv")
df = df[df["soup"].notna()]

count = CountVectorizer(stop_words="english")
countmatrix = count.fit_transform(df["soup"])

cosine_sim = cosine_similarity(countmatrix, countmatrix)

df = df.reset_index()
indices = pd.Series(df.index, index= df["title"])

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df[['title', 'poster-link', 'runtime', 'overview', 'vote_average']].iloc[movie_indices].values.tolist()



