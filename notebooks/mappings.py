from ratelimiter import RateLimiter
from tenacity import retry, stop_after_attempt, wait_fixed
import pickle
import requests
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from threading import Lock

q = Queue()
pbar_lock = Lock()
# Fetch the gene description from NCBI Entrez


@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.2))
@RateLimiter(max_calls=9, period=1)
def fetch_entrez_gene(id):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "gene",
        "id": id,
        "retmode": "json",
        "api_key": "08c3e5645f832a8ef99f034c2e9dd39a7d09"
    }
    response = requests.get(url, params=params)
    return response.json()["result"][id]["description"] if response.status_code == 200 else None

# Fetch the disease name from MESH


@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.2))
@RateLimiter(max_calls=9, period=1)
def fetch_mesh_descriptor(id):
    url = f"https://id.nlm.nih.gov/mesh/lookup/label?resource={id}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200 and response.json():
        return response.json()[0]
    else:
        return None

# Fetch the species scientific name from NCBI taxonomy


@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.2))
@RateLimiter(max_calls=9, period=1)
def fetch_ncbi_species(id):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=taxonomy&id={id}&api_key=08c3e5645f832a8ef99f034c2e9dd39a7d09"
    response = requests.get(url)
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        return root.find('.//Item[@Name="ScientificName"]').text
    else:
        return None

# Fetch the cell line from Cellosaurus


@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.2))
@RateLimiter(max_calls=9, period=1)
def fetch_cellosaurus(id):
    url = f"https://api.cellosaurus.org/cell-line/{id}"
    response = requests.get(url)
    if response.status_code == 200:
        return [value['value'] for name in response.json().get('Cellosaurus', {}).get('cell-line-list', []) for value in name.get('name-list', [])]
    else:
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
        return list(gene_set)
    else:
        return None


@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.2))
@RateLimiter(max_calls=9, period=1)
def fetch_dbsnp(rs_id):
    url = f"https://api.ncbi.nlm.nih.gov/variation/v0/beta/refsnp/{rs_id}&api_key=08c3e5645f832a8ef99f034c2e9dd39a7d09"
    response = requests.get(url)
    if response.status_code == 200:
        return parse_dbsnp(response.json())
    else:
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
# Convert keys to list for subscription
concept_keys = list(concept_vectors.keys())

# Fetch the descriptions
concept_descriptions = {concept: None for concept in concept_keys}


def update_concept_description(concept):
    concept_descriptions.update({concept: fetch_concept_description(concept)})


with tqdm(total=len(concept_descriptions)) as pbar:
    # max_workers can be adjusted
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(update_concept_description, concept)
                   for concept in concept_descriptions.keys()}

        for future in concurrent.futures.as_completed(futures):
            q.get()
            with pbar_lock:
                pbar.update()

# Save to pickle file
with open('/Users/danielgeorge/Documents/work/ml/bioconceptvec-explorer/bioconceptvec-explorer/datasets/concept_descriptions.pkl', 'wb') as f:
    pickle.dump(concept_descriptions, f)
