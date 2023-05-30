from tqdm import tqdm
from gensim.models import KeyedVectors
import os, sys, json, numpy as np


# word embedding
YOUR_BIOCONCEPTVEC_PATH = './embeddings/bioconceptvec_glove.bin'

# concept embedding
YOUR_JSON_PATH = './embeddings/concept_glove.json'

def load_embedding(path, binary):
    embedding = KeyedVectors.load_word2vec_format(path, binary=binary)
    print('embedding loaded from', path)
    return embedding


print('loading concept embedding...')
with open(YOUR_JSON_PATH) as json_file:  
    concept_vectors = json.load(json_file)
print('loaded', len(concept_vectors), 'concepts')

result = concept_vectors["Gene_2997"]
result += concept_vectors["Gene_3586"]

K = 10

print('calculating similarity...')
similarity = {}
for concept, vector in tqdm(concept_vectors.items()):
    similarity[concept] = np.dot(result, vector) / (np.linalg.norm(result) * np.linalg.norm(vector))

print(f'top {K} similar concepts:')
for i, (concept, sim) in enumerate(sorted(similarity.items(), key=lambda x: x[1], reverse=True)):
    if i == K:
        break
    print(f"{i+1}. {concept} ({sim})")