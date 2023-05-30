from tqdm import tqdm
import numpy as np
import re
import os
import json
import modal
from fastapi import FastAPI

# todo fix the code and deploy as api


# app = FastAPI()
# image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")
# stub = modal.Stub("landingpage-autobuild", image=image)
# mounts = [
#     modal.Mount.from_local_file("../embeddings/", remote_path="/root/embeddings/"),
# ]

# @stub.function(mounts=mounts)
# @web_endpoint(method="GET", path="/")
def component_gen(
    equation: str,
) -> dict:
    LHS, RHS = parse_equation(equation)
    LHS, RHS = solve_for_x(LHS, RHS)

    # load concept embedding
    with open("../embeddings/concept_glove.json") as json_file:
        concept_vectors = json.load(json_file)
    
    # compute x vector
    result = np.zeros(np.array(concept_vectors["Gene_2997"]).shape)
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
        raise ValueError("Solved equation does not contain x isolated on one side")
    
    # compute similarity between x vector and all other vectors
    similarity = {}
    for concept, vector in tqdm(concept_vectors.items()):
        similarity[concept] = np.dot(result, vector) / (np.linalg.norm(result) * np.linalg.norm(vector))
    
    # return top 10 most similar concepts
    K = 10
    top_concepts = {}
    for i, (concept, sim) in enumerate(sorted(similarity.items(), key=lambda x: x[1], reverse=True)):
        if i == K:
            break
        top_concepts[concept] = sim
    
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


def main():
    top = component_gen("Gene_2997 = Gene_3586 + x")
    for key, value in top.items():
        print(key, value)

if __name__ == "__main__":
    main()
