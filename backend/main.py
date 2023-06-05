import streamlit as st
import json
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

# load concept embedding for all API calls
st.write("Cold start - loading concept embeddings...")
with open("../embeddings/concept_glove.json") as json_file:
    concept_vectors = json.load(json_file)
    concept_keys = list(concept_vectors.keys())
    concept_values = np.array(list(concept_vectors.values()), dtype=np.float32)
st.write("Done!")


def compute_expression(expression: str, k: int = 10, useCosineSimilarity: bool = True) -> dict:
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
        st.write(f"Variable: {variable} | Sign: {sign}")
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
    top_concepts = dict(sorted(top_concepts.items(),
                        key=lambda item: item[1], reverse=True))
    return top_concepts


def autosuggest(query: str, limit: int) -> list:
    # filter concept vectors based on whether query is a substring
    query = query.lower()
    lower_concept_vectors = map(lambda x: x.lower(), concept_vectors.keys())
    result = [concept for concept in lower_concept_vectors if query in concept]
    return result[:limit]


def get_similar_concepts(concept_query: str, k: int) -> list:
    concept = concept_vectors[concept_query]
    similarities = cosine_similarity(concept_values, [concept]).flatten()
    top_concepts = {}
    for concept, similarity in zip(concept_vectors.keys(), similarities):
        top_concepts[concept] = similarity
    top_concepts = dict(sorted(top_concepts.items(),
                        key=lambda item: item[1], reverse=True)[:k])
    return top_concepts



st.title("BioConceptVec Exploration App")

    # Initialize session state for the query if it doesn't exist yet
    if "query" not in st.session_state:
        st.session_state["query"] = ""

    if option == "Compute expression":
        st.session_state.query = st.text_input(
            "Enter your expression", st.session_state.query
        )
        suggestions = autosuggest(
            st.session_state.query, 10
        )  # Adjust the limit based on your needs
        st.write("Suggestions:", suggestions)

if option == "Compute expression":
    st.session_state.query = st.text_input(
        "Enter your expression", st.session_state.query)
    # Adjust the limit based on your needs
    suggestions = autosuggest(st.session_state.query, 10)
    st.write("Suggestions:", suggestions)

    k = st.number_input(
        "Enter the number of top similar concepts", min_value=1, value=10, step=1)
    if st.button("Compute"):
        result = compute_expression(st.session_state.query, k)
        st.write(result)
elif option == "Autosuggest":
    st.session_state.query = st.text_input(
        "Enter your query", st.session_state.query)
    suggestions = autosuggest(st.session_state.query, 10)
    st.write("Suggestions:", suggestions)

    limit = st.number_input("Enter the limit", min_value=1, value=10, step=1)
    if st.button("Suggest"):
        result = autosuggest(st.session_state.query, limit)
        st.write(result)
elif option == "Get similar concepts":
    st.session_state.query = st.text_input(
        "Enter your concept query", st.session_state.query)
    suggestions = autosuggest(st.session_state.query, 10)
    st.write("Suggestions:", suggestions)

    k = st.number_input(
        "Enter the number of top similar concepts", min_value=1, value=10, step=1)
    if st.button("Get"):
        result = get_similar_concepts(st.session_state.query, k)
        st.write(result)
