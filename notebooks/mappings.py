import concurrent
from concurrent.futures import ThreadPoolExecutor
import pickle
import requests
import xml.etree.ElementTree as ET
from threading import Semaphore, Thread, Lock
import time
import shelve
import json
from tqdm import tqdm
from queue import Queue

# The semaphore to limit the number of simultaneous API calls
sem = Semaphore(3)

# Lock for updating the progress bar
pbar_lock = Lock()

# Queue for thread-safe progress updates
q = Queue()

# Fetch the gene description from NCBI Entrez


def fetch_entrez_gene(id):
    time.sleep(0.33)
    try:
        with sem:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params = {
                "db": "gene",
                "id": id,
                "retmode": "json"
            }
            response = requests.get(url, params=params)
            return response.json()["result"][id]["description"] if response.status_code == 200 else None
    except Exception as e:
        print(f"An error occurred in fetch_entrez_gene: {id}")
        return None

# Fetch the disease name from MESH


def fetch_mesh_descriptor(id):
    time.sleep(0.33)
    try:
        with sem:
            url = f"https://id.nlm.nih.gov/mesh/lookup/label?resource={id}"
            headers = {"Accept": "application/json"}
            response = requests.get(url, headers=headers)
            if response.status_code == 200 and response.json():
                return response.json()[0]
            else:
                return None
    except Exception as e:
        print(f"An error occurred in fetch_mesh_descriptor: {e}")
        return None

# Fetch the species scientific name from NCBI taxonomy


def fetch_ncbi_species(id):
    time.sleep(0.33)
    try:
        with sem:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=taxonomy&id={id}"
            response = requests.get(url)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                return root.find('.//Item[@Name="ScientificName"]').text
            else:
                return None
    except Exception as e:
        print(f"An error occurred in fetch_ncbi_species: {e}")
        return None

# Fetch the cell line from Cellosaurus


def fetch_cellosaurus(id):
    time.sleep(0.33)
    try:
        with sem:
            url = f"https://api.cellosaurus.org/cell-line/{id}"
            response = requests.get(url)
            if response.status_code == 200:
                return [value['value'] for name in response.json().get('Cellosaurus', {}).get('cell-line-list', []) for value in name.get('name-list', [])]
            else:
                return None
    except Exception as e:
        print(f"An error occurred in fetch_cellosaurus: {e}")
        return None

# Fetch the SNP from dbSNP


def parse_dbsnp(dbsnp_json):
    if dbsnp_json:
        gene_set = set()
        if 'primary_snapshot_data' in dbsnp_json:
            if 'allele_annotations' in dbsnp_json['primary_snapshot_data']:
                for alle_annotation in dbsnp_json["primary_snapshot_data"]["allele_annotations"]:
                    if 'assembly_annotation' in alle_annotation:
                        for assembly_annotation in alle_annotation["assembly_annotation"]:
                            if 'genes' in assembly_annotation:
                                for gene in assembly_annotation['genes']:
                                    if 'name' in gene:
                                        gene_set.add(gene['name'])
        snps_list = []
        if 'primary_snapshot_data' in dbsnp_json:
            if 'placements_with_allele' in dbsnp_json['primary_snapshot_data']:
                for placements_with_allele in dbsnp_json["primary_snapshot_data"]["placements_with_allele"]:
                    if 'alleles' in placements_with_allele:
                        snps_list.append(placements_with_allele['alleles'])
        # return {
        #     "chromosome": dbsnp_json["refsnp_id"],
        #     "snps": snps_list,
        #     "gene": list(gene_set)
        # }
        return list(gene_set)
    else:
        return None


def fetch_dbsnp(rs_id):
    time.sleep(0.33)
    try:
        with sem:
            url = f"https://api.ncbi.nlm.nih.gov/variation/v0/beta/refsnp/{rs_id}"
            response = requests.get(url)
            if response.status_code == 200:
                return parse_dbsnp(response.json())
            else:
                return None
    except Exception as e:
        print(f"An error occurred in fetch_dbsnp: {e}")
        return None

# Fetch the concept description


def fetch_concept_description(concept_id):
    try:
        # When finished, put a dummy item in the queue to signal one task's completion
        q.put(None)
        concept_type, identifier = concept_id.split('_', 1)
        result = None
        if concept_type == "Disease":
            if identifier.startswith("MESH"):
                identifier = identifier[identifier.find('_')+1:]
                result = fetch_mesh_descriptor(identifier)
            else:
                result = identifier[identifier.find('_')+1:].replace('_', ' ')
        elif concept_type == "Gene":
            # If there are any more '_' in the identifier, split and take the first part
            if '_' in identifier:
                identifier = identifier.split('_')[0]
            result = fetch_entrez_gene(identifier)
        elif concept_type == "Species":
            result = fetch_ncbi_species(identifier)
        elif concept_type == "CellLine":
            result = fetch_cellosaurus(identifier)
        elif concept_type == "ProteinMutation":
            identifier = identifier[identifier.rfind('_')+1:]
            result = fetch_dbsnp(identifier)
        elif concept_type == "SNP":
            time.sleep(0.1)
            identifier = identifier[2:]
            result = fetch_dbsnp(identifier)
        elif concept_type == "Chemical":
            identifier = identifier[identifier.find('_')+1:]
            result = fetch_mesh_descriptor(identifier)
        elif concept_type == "DNAMutation":
            result = concept_type + ' ' + identifier.replace('_', ' ')
        elif concept_type == "DomainMotif":
            result = concept_type + ' ' + identifier.replace('_', ' ')
        else:
            return None

        if result is None:
            return concept_type + ' ' + identifier.replace('_', ' ')
        else:
            return result

    except Exception as e:
        print(f"An error occurred in fetch_concept_description: {e}")
        return None


YOUR_JSON_PATH = '/Users/danielgeorge/Documents/work/ml/bioconceptvec-explorer/bioconceptvec-explorer/embeddings/concept_glove.json'
# Load concepts from JSON
with open(YOUR_JSON_PATH) as json_file:
    concept_vectors = json.load(json_file)
print('load', len(concept_vectors), 'concepts')

# Convert keys to list for subscription
concept_keys = list(concept_vectors.keys())

# Fetch the descriptions
concept_descriptions = {concept: None for concept in concept_keys}


def update_concept_description(concept):
    concept_descriptions.update({concept: fetch_concept_description(concept)})


with tqdm(total=len(concept_descriptions)) as pbar:
    # max_workers can be adjusted
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(update_concept_description, concept)
                   for concept in concept_descriptions.keys()}

        for future in concurrent.futures.as_completed(futures):
            q.get()
            with pbar_lock:
                pbar.update()

# Save to pickle file
with open('/Users/danielgeorge/Documents/work/ml/bioconceptvec-explorer/bioconceptvec-explorer/datasets/concept_descriptions.pkl', 'wb') as f:
    pickle.dump(concept_descriptions, f)
