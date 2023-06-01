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
    modal.Mount.from_local_dir(
        "./embeddings/", remote_path="/root/embeddings/"),
]

image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

stub = modal.Stub("bioconceptvec", image=image, mounts=mounts)


@stub.function()
@modal.web_endpoint(method="GET")
def solve_equation(
    equation: str,
    k: int = 10,
    concept_vectors: list = None,
    concept_values: np.ndarray = None,
    useCosineSimilarity: bool = True,
) -> dict:
    LHS, RHS = parse_equation(equation)
    LHS, RHS = solve_for_x(LHS, RHS)

    # load concept embedding
    if concept_values is None or concept_vectors is None:
        print("Loading concept embeddings...")
        with open("./embeddings/concept_glove.json") as json_file:
            concept_vectors = json.load(json_file)
            concept_values = np.array(
                list(concept_vectors.values()), dtype=np.float32)

    # compute x vector
    result = np.zeros(np.array(concept_values[0]).shape, dtype=np.float32)
    if "x" in RHS["positive"]:
        for variable in LHS["positive"]:
            result += concept_vectors[variable]
        for variable in LHS["negative"]:
            result -= concept_vectors[variable]
    elif "x" in LHS["positive"]:
        for variable in RHS["positive"]:
            result -= concept_vectors[variable]
        for variable in RHS["negative"]:
            result += concept_vectors[variable]
    else:
        raise ValueError(
            "Solved equation does not contain x isolated on one side")

    top_concepts = {}
    similarities = None

    if useCosineSimilarity:
        # compute similarity between x vector and all other vectors
        similarities = cosine_similarity(concept_values, [result]).flatten()
    else:
        # compute distance between x vector and all other vectors
        similarities = np.linalg.norm(concept_values - result, axis=1)

    # return top k most similar concepts
    for concept, similarity in zip(concept_vectors.keys(), similarities):
        top_concepts[concept] = similarity
    top_concepts = dict(
        sorted(top_concepts.items(),
               key=lambda item: item[1], reverse=useCosineSimilarity)[:k]
    )

    return top_concepts


def solve_for_x(LHS, RHS):
    # solve for x in terms of other variables
    if "x" in LHS["positive"]:
        for variable in LHS["positive"]:
            if variable != "x":
                RHS["negative"].append(variable)
        for variable in LHS["negative"]:
            RHS["positive"].append(variable)
        LHS["positive"] = ["x"]
        LHS["negative"] = []
    elif "x" in LHS["negative"]:
        for variable in LHS["positive"]:
            RHS["positive"].append(variable)
        for variable in LHS["negative"]:
            if variable != "x":
                RHS["negative"].append(variable)
        LHS["positive"] = ["x"]
        LHS["negative"] = []
        RHS["positive"], RHS["negative"] = RHS["negative"], RHS["positive"]
    elif "x" in RHS["positive"]:
        for variable in RHS["positive"]:
            if variable != "x":
                LHS["negative"].append(variable)
        for variable in RHS["negative"]:
            LHS["positive"].append(variable)
        RHS["positive"] = ["x"]
        RHS["negative"] = []
    elif "x" in RHS["negative"]:
        for variable in RHS["positive"]:
            LHS["positive"].append(variable)
        for variable in RHS["negative"]:
            if variable != "x":
                LHS["negative"].append(variable)
        RHS["positive"] = ["x"]
        RHS["negative"] = []
        LHS["positive"], LHS["negative"] = LHS["negative"], LHS["positive"]
    else:
        raise ValueError("Equation does not contain x")

    return LHS, RHS


def parse_equation(equation):
    # Remove any whitespace from the equation
    equation = equation.replace(" ", "")

    # Split the equation into left-hand side (LHS) and right-hand side (RHS)
    lhs, rhs = equation.split("=")
    if lhs[0] != "-":
        lhs = "+" + lhs
    if rhs[0] != "-":
        rhs = "+" + rhs

    # Regular expression pattern to match operators and variables
    pattern = r"([\+\-]?)([a-zA-Z\_0-9]+)"

    # Parse LHS
    lhs_matches = re.findall(pattern, lhs)
    LHS = {"positive": [], "negative": []}
    for match in lhs_matches:
        sign, variable = match
        if sign == "+":
            LHS["positive"].append(variable)
        elif sign == "-":
            LHS["negative"].append(variable)

    # Parse RHS
    rhs_matches = re.findall(pattern, rhs)
    RHS = {"positive": [], "negative": []}
    for match in rhs_matches:
        sign, variable = match
        if sign == "+":
            RHS["positive"].append(variable)
        elif sign == "-":
            RHS["negative"].append(variable)

    return LHS, RHS


@stub.function()
def find_equations(sim_threshold: int = 0.95, n: int = 1000):
    # load concept embedding
    print("Loading concept embeddings...")
    with open("./embeddings/concept_glove.json") as json_file:
        concept_vectors = json.load(json_file)
    concept_keys = list(concept_vectors.keys())
    concept_values = np.array(list(concept_vectors.values()), dtype=np.float32)

    # pick random concepts to fill equations of the form
    # # a - b + c = x

    # randomly pick n triplets of concepts for a, b, c
    print("Generating equations...")
    concepts = list(concept_vectors.keys())
    equations = []
    for _ in range(n):
        a, b, c = random.sample(concepts, 3)
        equations.append(f"{a} - {b} + {c} = x")

    # for each, compute similarity between x vector and all other vectors
    print("Solving equations...")
    good_equations = []
    for equation in tqdm(equations):
        concept, sim = solve_equation(
            equation,
            k=1,
            concept_vectors=concept_vectors,
            concept_values=concept_values,
        ).popitem()
        if sim > sim_threshold:
            good_equations.append((equation, concept, sim))
            print(
                f"Equation: {equation} | Solution: {concept} | Similarity: {sim}")

    # now take the top 10% of the good equations and mutate them by
    # # finding most similar 10 concepts to each variable
    # # swapping it with one of those at random
    # # adding it to the new population and recompute similarity


def main():
    # top = solve_equation("Gene_2997 - Chemical_MESH_C114286 = Gene_3586 + x")
    # for key, value in top.items():
    #     print(key, value)
    find_equations()


if __name__ == "__main__":
    main()
