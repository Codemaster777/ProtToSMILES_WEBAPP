import os
import time
import numpy as np
from flask import Flask, render_template, request, send_file
from rdkit import Chem
from transformers import AutoModel
from bio_embeddings.embed import ProtTransBertBFDEmbedder
from modelstrc import CVanilla_RNN_Builder, get_mol_from_graph_list

# Define absolute paths inside the container
BASE_DIR = "/app"
BIO_MODEL_DIR = os.path.join(BASE_DIR, "modelsBioembed")  # Bio-embeddings directory
CVN_MODEL_DIR = os.path.join(BASE_DIR, "models_folder")  # CVanilla_RNN_Builder directory
UPLOAD_FOLDER = os.path.join(BASE_DIR, "Samples")  # Samples directory

# Ensure directories exist
os.makedirs(BIO_MODEL_DIR, exist_ok=True)
os.makedirs(CVN_MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set environment variables for temporary storage
os.environ["TMPDIR"] = BIO_MODEL_DIR
os.environ["TEMP"] = BIO_MODEL_DIR
os.environ["TMP"] = BIO_MODEL_DIR

app = Flask(__name__)

# Load pre-trained model if not already downloaded
model_path = os.path.join(BIO_MODEL_DIR, "pytorch_model.bin")
if not os.path.exists(model_path):
    print("Downloading ProtTrans-BERT-BFD model...")
    AutoModel.from_pretrained("Rostlab/prot_bert_bfd", low_cpu_mem_usage=True).save_pretrained(BIO_MODEL_DIR)

# Load bio-embedding model
try:
    embedder = ProtTransBertBFDEmbedder(model_directory=BIO_MODEL_DIR)
except Exception as e:
    print(f"Error loading ProtTrans-BERT-BFD model: {e}")
    embedder = None

def generate_bio_embeddings(sequence):
    """Generate bio-embeddings for a given protein sequence."""
    if embedder is None:
        return None
    try:
        embedding_protein = embedder.embed(sequence)
        embedding_per_protein = embedder.reduce_per_protein(embedding_protein)
        return np.array(embedding_per_protein).reshape(1, -1)
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

def generate_smiles(sequence, n_samples=100):
    """Generate SMILES from a protein sequence."""
    start_time = time.time()

    protein_embedding = generate_bio_embeddings(sequence)
    if protein_embedding is None:
        return None, "Embedding generation failed!"

    # Load trained CVanilla_RNN_Builder model
    model = CVanilla_RNN_Builder(CVN_MODEL_DIR, gpu_id=None)

    # Generate molecular graphs
    samples = model.sample(n_samples, c=protein_embedding[0], output_type='graph')
    valid_samples = [sample for sample in samples if sample is not None]

    # Convert to SMILES
    smiles_list = [
        Chem.MolToSmiles(mol) for mol in get_mol_from_graph_list(valid_samples, sanitize=True) if mol is not None
    ]

    if not smiles_list:
        return None, "No valid SMILES generated!"

    # Save to file
    filename = os.path.join(UPLOAD_FOLDER, "SMILES_GENERATED.txt")
    with open(filename, "w") as file:
        file.write("\n".join(smiles_list))

    elapsed_time = time.time() - start_time
    return filename, elapsed_time

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        sequence = request.form["sequence"].strip()
        if not sequence:
            return render_template("index.html", message="Please enter a valid sequence.")

        file_path, result = generate_smiles(sequence)
        if file_path is None:
            return render_template("index.html", message=f"Error: {result}")

        return render_template("index.html", message="SMILES generated successfully!", file_path=file_path, time_taken=result)

    return render_template("index.html")

@app.route("/download")
def download_file():
    file_path = os.path.join(UPLOAD_FOLDER, "SMILES_GENERATED.txt")
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
