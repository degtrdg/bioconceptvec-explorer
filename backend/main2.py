import streamlit as st
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load concept_descriptions function


@st.cache_data
def load_concept_descriptions():
    concept_descriptions = pd.read_pickle(
        '/Users/danielgeorge/Documents/work/ml/bioconceptvec-explorer/bioconceptvec-explorer/mappings/concept_descriptions.pkl')
    return concept_descriptions
# Load sentence_embeddings function


@st.cache_data
def load_sentence_embeddings():
    sentence_embeddings = np.load(
        '/Users/danielgeorge/Documents/work/ml/bioconceptvec-explorer/bioconceptvec-explorer/mappings/description_embeddings.npy')
    return sentence_embeddings
# Load sentences function


@st.cache_data
def load_sentences():
    with open('/Users/danielgeorge/Documents/work/ml/bioconceptvec-explorer/bioconceptvec-explorer/notebooks/sentences.txt') as f:
        sentences = f.readlines()
    return sentences


# Load the necessary data
concept_descriptions = load_concept_descriptions()
sentence_embeddings = load_sentence_embeddings()
sentences = load_sentences()

# Load the model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Initialize the index
d = sentence_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(sentence_embeddings)


def process_input(user_input):
    k = 8
    xq = model.encode([user_input])
    D, I = index.search(xq, k)
    options = [f'{i}: {sentences[i]}' for i in I[0]]
    return options


def select_option(options):
    selected_option = st.selectbox("Select a similar concept:", options)
    if selected_option:
        st.write("You selected:", selected_option)
    return selected_option


# Set up the Streamlit page
st.title("Concept Explorer")

# Get the user's input
user_input = st.text_input("Enter a concept:")

if user_input:
    options = process_input(user_input)
    if options:
        option = select_option(options)
        if option:
            # st.write(concept_descriptions[int(option.split(':')[0])])
            start_index = option.find(':') + 1
            end_index = option.find('|')
            extracted_string = option[start_index:end_index].strip()
            st.write(extracted_string)
