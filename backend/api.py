import time
import openai
import pandas as pd
import pickle
import random
from tqdm import tqdm
import numpy as np
import re
import os
import json
import modal
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity

import os
import json
import openai
import dotenv


def load_openai_key(path):
    dotenv.load_dotenv(path)
    openai.api_key = os.getenv("OPENAI_API_KEY")


def build_messages_from_file(path, prompt):
    messages = json.load(open(path, "r"))
    messages.append({"role": "user", "content": prompt})
    return messages


def get_prompt(query: str):
    return f"""
        What does this mean analogically? I found this by doing equations with vector embeddings.
        This is similar to how King - Man + Woman = Queen for word2vec. I'm trying to reason why this makes sense.

        {query}

        Really try to think outside the box to find why this could be reasonable. Use this as a generative way to help think of biological hypotheses.
        """


def gpt(prompt):
    load_openai_key("./.env")
    messageList = [
        {
            "role": "system",
            "content": "You are a helpful chatbot that helps people understand biology.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messageList
    )

    feature_string = completion.choices[0].message.content

    return feature_string


# load concept embedding for all API calls
print("Cold start - loading concept embeddings...")
with open("./embeddings/concept_glove.json") as json_file:
    concept_vectors = json.load(json_file)
    concept_keys = list(concept_vectors.keys())
    concept_values = np.array(list(concept_vectors.values()), dtype=np.float32)

print("loading concept descriptions...")
with open("./mappings/concept_descriptions.pkl", "rb") as f:
    concept_descriptions = pickle.load(f)
    rev_concept_descriptions = {}
    for key, value in tqdm(concept_descriptions.items()):
        if type(value) == list and len(value) == 0:
            continue
        elif type(value) == list and len(value) > 0:
            rev_concept_descriptions[value[0]] = key
        else:
            rev_concept_descriptions[value] = key

print("Done!")


@stub.function()
@modal.web_endpoint(method="GET")
def compute_expression(
    expression: list,
    k: int = 10,
    useCosineSimilarity: bool = True,
) -> dict:
    # print(f"Computing expression: {expression}")

    if expression[0] != "-" and expression[0] != "+":
        expression = ["+", *expression]

    # split expression into groups of 2 (sign, concept)
    matches = [expression[i: i + 2] for i in range(0, len(expression), 2)]
    # compute x vector
    result = np.zeros(np.array(concept_values[0]).shape, dtype=np.float32)
    for match in matches:
        sign, variable = match
        # print(f"Variable: {variable} | Sign: {sign}")
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
        similarities = np.linalg.norm(
            concept_values - result, axis=1).flatten()

    # get index of top k similarities
    top_k_indices = np.argpartition(similarities, -k)[-k:]

    # get top k most similar concepts as a dict
    top_concepts = {concept_keys[i]: float(
        similarities[i]) for i in top_k_indices}
    top_concepts = dict(
        sorted(top_concepts.items(), key=lambda item: item[1], reverse=True)
    )
    return top_concepts


@stub.function()
@modal.web_endpoint(method="GET")
def autosuggest(query: str, limit: int) -> list:
    # filter concept vectors based on whether query is a substring
    query = query.lower()
    descs = list(concept_descriptions.values())
    for i in range(len(descs)):
        if type(descs[i]) == list and len(descs[i]) > 0:
            descs[i] = descs[i][0]
        elif type(descs[i]) == list and len(descs[i]) == 0:
            descs[i] = ""

    descs = [i for i in descs if i is not None and i != ""]
    lower_concept_descs = map(lambda x: x.lower(), descs)
    result = [concept for concept in lower_concept_descs if query in concept]
    return result[:limit]


@stub.function()
@modal.web_endpoint(method="GET")
def get_similar_concepts(concept_query: str, k: int) -> list:
    # convert from concept description to concept id
    if ";" in concept_query:
        concept_query = concept_query.split(";")[0]
    concept_query = rev_concept_descriptions[concept_query]
    concept = concept_vectors[concept_query]
    similarities = cosine_similarity(concept_values, [concept]).flatten()
    top_concepts = {}
    for concept, similarity in zip(concept_vectors.keys(), similarities):
        top_concepts[concept] = similarity
    top_concepts = dict(
        sorted(top_concepts.items(),
               key=lambda item: item[1], reverse=True)[:k]
    )
    return top_concepts


@stub.function()
@modal.web_endpoint(method="GET")
def free_var_search(term: str, sim_threshold=0.7, n=100, top_k=3, use_gpt=False):
    term_vec = concept_vectors[term]
    expressions = []

    # randomly pick 1000 pairs of concepts for b, c
    concepts = list(concept_vectors.keys())
    equations = []
    for _ in range(n):
        b, c = random.sample(concepts, 2)
        equations.append([term, "+", b, "-", c])

    print("Solving equations...")
    good_equations = []
    for equation in tqdm(equations):
        concept, sim = compute_expression(
            equation,
            k=1,
        ).popitem()
        if sim > sim_threshold and concept not in equation:
            print(
                f"Equation: {equation} | Concept: {concept} | Similarity: {sim}")
            good_equations.append((equation, concept, sim))
            print(
                f"Expression: {equation} | Solution: {concept} | Similarity: {sim}")

    df = pd.DataFrame(good_equations, columns=[
                      "Equation", "Concept", "Similarity"])
    # Sort by similarity
    df = df.sort_values(by=["Similarity"], ascending=False)
    df = df.reset_index(drop=True)
    # Pick top k
    df = df[:top_k]

    # now we use gpt to generate a rationale for each equation using the prompt
    if use_gpt:
        rationales = []
        for row in tqdm(df.iterrows()):
            mapped_eq = row[1]["Equation_mapped"]
            prompt = get_prompt(mapped_eq)
            rationales.append(gpt(prompt))
            time.sleep(0.5)

        df["Rationale"] = rationales

    df.to_csv("results.csv", index=False)

    return df


@stub.function()
@modal.web_endpoint(method="GET")
def heartbeat():
    return "OK"


def main():
    # print(
    #     compute_expression(["2-iodobenzylguanidine", "+", "adrenoceptor beta 1"], k=10)
    # )
    free_var_search("Gene_6406_29106")


if __name__ == "__main__":
    main()
