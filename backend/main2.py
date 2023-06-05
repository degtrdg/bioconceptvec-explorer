import streamlit as st
import numpy as np
import faiss
import pandas as pd
import streamlit_pandas as sp
from sentence_transformers import SentenceTransformer
from api import free_var_search

# Load concept_descriptions function


# load concept embedding for all API calls
st.write("Cold start - loading concept embeddings...")


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

st.write("Done!")


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
st.title("BioConceptVec Exploration App")

# Get the user's input
user_input = st.text_input("Enter a concept:")

if user_input:
    options = process_input(user_input)
    if options:
        option = select_option(options)
        if option:
            start_index = option.find(':') + 1
            end_index = option.find('|')
            extracted_string = option[start_index:end_index].strip()
            st.write(extracted_string)
            # Make an input box from 0.0 to 1.0 by increments of 0.1 multiselect
            threshold = st.multiselect(
                'Select a threshold:', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            if threshold:
                threshold = threshold[0]
                free_var_search(extracted_string, threshold,
                                use_gpt=True, top_k=10)
                import streamlit as st
                import pandas as pd

                # Load the CSV file
                data = pd.read_csv('results.csv')

                # Display a download button
                st.download_button(label='Download CSV', data=data.to_csv(
                ), file_name='res.csv', mime='text/csv')

                # Show the dataframe
                # sp.write(data)
