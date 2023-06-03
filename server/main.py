import random
from tqdm import tqdm
import numpy as np
import re
import os
import json
import modal
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity

mounts = [
    modal.Mount.from_local_dir("./embeddings/", remote_path="/root/embeddings/"),
]

image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

stub = modal.Stub("bioconceptvec", image=image, mounts=mounts)

# load concept embedding for all API calls
print("Cold start - loading concept embeddings...")
with open("./embeddings/concept_glove.json") as json_file:
    concept_vectors = json.load(json_file)
    concept_keys = list(concept_vectors.keys())
    concept_values = np.array(list(concept_vectors.values()), dtype=np.float32)
print("Done!")


@stub.function()
@modal.web_endpoint(method="GET")
def compute_expression(
    expression: str,
    k: int = 10,
    useCosineSimilarity: bool = True,
) -> dict:
    print(f"Computing expression: {expression}")
    # remove whitespace
    expression = expression.replace(" ", "")
    if expression[0] != "-":
        expression = "+" + expression

    # regex to match operators and operands
    pattern = r"([\+\-]?)([a-zA-Z\_0-9]+)"
    matches = re.findall(pattern, expression)

    # compute x vector
    result = np.zeros(np.array(concept_values[0]).shape, dtype=np.float32)
    for match in matches:
        sign, variable = match
        print(f"Variable: {variable} | Sign: {sign}")
        if sign == "-":
            result -= concept_vectors[variable]
        elif sign == "+":
            result += concept_vectors[variable]
        else:
            raise ValueError(f"Invalid operator: {sign}")

    similarities = None
    if useCosineSimilarity:
        # compute similarity between x vector and all other vectors
        similarities = cosine_similarity(concept_values, [result]).flatten()
    else:
        # compute distance between x vector and all other vectors
        similarities = np.linalg.norm(concept_values - result, axis=1).flatten()

    # get index of top k similarities
    top_k_indices = np.argpartition(similarities, -k)[-k:]

    # get top k most similar concepts as a dict
    top_concepts = {concept_keys[i]: float(similarities[i]) for i in top_k_indices}
    top_concepts = dict(
        sorted(top_concepts.items(), key=lambda item: item[1], reverse=True)
    )
    return top_concepts


@stub.function()
@modal.web_endpoint(method="GET")
def autosuggest(query: str, limit: int) -> list:
    # filter concept vectors based on whether query is a substring
    query = query.lower()
    lower_concept_vectors = map(lambda x: x.lower(), concept_vectors.keys())
    result = [concept for concept in lower_concept_vectors if query in concept]
    return result[:limit]


@stub.function()
@modal.web_endpoint(method="GET")
def get_similar_concepts(concept_query: str, k: int) -> list:
    concept = concept_vectors[concept_query]
    similarities = cosine_similarity(concept_values, [concept]).flatten()
    top_concepts = {}
    for concept, similarity in zip(concept_vectors.keys(), similarities):
        top_concepts[concept] = similarity
    top_concepts = dict(
        sorted(top_concepts.items(), key=lambda item: item[1], reverse=True)[:k]
    )
    return top_concepts


def find_equations(sim_threshold: int = 0.95, n: int = 1000):
    # pick random concepts to fill equations of the form
    # # a - b + c = x

    # randomly pick n triplets of concepts for a, b, c
    print("Generating equations...")
    concepts = list(concept_vectors.keys())
    equations = []
    for _ in range(n):
        a, b, c = random.sample(concepts, 3)
        equations.append(f"{a} - {b} + {c}")

    # for each, compute similarity between x vector and all other vectors
    print("Solving equations...")
    good_equations = []
    for equation in tqdm(equations):
        concept, sim = compute_expression(
            equation,
            k=1,
            concept_vectors=concept_vectors,
            concept_values=concept_values,
        ).popitem()
        if sim > sim_threshold:
            good_equations.append((equation, concept, sim))
            print(f"Expression: {equation} | Solution: {concept} | Similarity: {sim}")

    # now take the top 10% of the good equations and mutate them by
    # # finding most similar 10 concepts to each variable
    # # swapping it with one of those at random
    # # adding it to the new population and recompute similarity


def main():
    print("Species_9615")
    print(compute_expression("Gene_91624-Gene_341346", k=10))


if __name__ == "__main__":
    main()
