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

print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG] BIO_MODEL_DIR: {BIO_MODEL_DIR}")
print(f"[DEBUG] CVN_MODEL_DIR: {CVN_MODEL_DIR}")
print(f"[DEBUG] UPLOAD_FOLDER: {UPLOAD_FOLDER}")

# Ensure directories exist
os.makedirs(BIO_MODEL_DIR, exist_ok=True)
os.makedirs(CVN_MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set environment variables for temporary storage
os.environ["TMPDIR"] = BIO_MODEL_DIR
os.environ["TEMP"] = BIO_MODEL_DIR
os.environ["TMP"] = BIO_MODEL_DIR

print("[DEBUG] Environment variables set for TMPDIR, TEMP, and TMP")

app = Flask(__name__)

# Load pre-trained model if not already downloaded
model_path = os.path.join(BIO_MODEL_DIR, "pytorch_model.bin")
print(f"[DEBUG] Checking model path: {model_path}")
if not os.path.exists(model_path):
    print("[DEBUG] Model not found. Downloading ProtTrans-BERT-BFD model...")
    AutoModel.from_pretrained("Rostlab/prot_bert_bfd", low_cpu_mem_usage=True).save_pretrained(BIO_MODEL_DIR)
else:
    print("[DEBUG] Model already exists.")

# Load bio-embedding model
try:
    print("[DEBUG] Loading ProtTrans-BERT-BFD embedder...")
    embedder = ProtTransBertBFDEmbedder(model_directory=BIO_MODEL_DIR)
    print("[DEBUG] Embedder loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading ProtTrans-BERT-BFD model: {e}")
    embedder = None

def generate_bio_embeddings(sequence):
    """Generate bio-embeddings for a given protein sequence."""
    print(f"[DEBUG] Generating bio-embeddings for sequence: {sequence}")
    if embedder is None:
        print("[ERROR] Embedder is None! Returning None.")
        return None
    try:
        embedding_protein = embedder.embed(sequence)
        embedding_per_protein = embedder.reduce_per_protein(embedding_protein)
        print("[DEBUG] Bio-embeddings generated successfully!")
        return np.array(embedding_per_protein).reshape(1, -1)
    except Exception as e:
        print(f"[ERROR] Embedding Error: {e}")
        return None

def generate_smiles(sequence, n_samples=100):
    """Generate SMILES from a protein sequence."""
    print(f"[DEBUG] Generating SMILES for sequence: {sequence}")
    start_time = time.time()
    
    protein_embedding = generate_bio_embeddings(sequence)
    if protein_embedding is None:
        print("[ERROR] Failed to generate embeddings!")
        return None, "Embedding generation failed!"

    # Load trained CVanilla_RNN_Builder model
    print("[DEBUG] Loading CVanilla_RNN_Builder model...")
    model = CVanilla_RNN_Builder(CVN_MODEL_DIR, gpu_id=None)
    print("[DEBUG] Model loaded successfully!")

    # Generate molecular graphs
    print("[DEBUG] Generating molecular graphs...")
    samples = model.sample(n_samples, c=protein_embedding[0], output_type='graph')
    valid_samples = [sample for sample in samples if sample is not None]
    print(f"[DEBUG] Number of valid samples: {len(valid_samples)}")

    # Convert to SMILES
    smiles_list = [
        Chem.MolToSmiles(mol) for mol in get_mol_from_graph_list(valid_samples, sanitize=True) if mol is not None
    ]
    print(f"[DEBUG] Number of valid SMILES: {len(smiles_list)}")

    if not smiles_list:
        print("[ERROR] No valid SMILES generated!")
        return None, "No valid SMILES generated!"

    # Save to file
    filename = os.path.join(UPLOAD_FOLDER, "SMILES_GENERATED.txt")
    print(f"[DEBUG] Saving SMILES to file: {filename}")
    with open(filename, "w") as file:
        file.write("\n".join(smiles_list))

    elapsed_time = time.time() - start_time
    print(f"[DEBUG] SMILES generation completed in {elapsed_time:.2f} seconds.")
    return filename, elapsed_time

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        sequence = request.form["sequence"].strip()
        print(f"[DEBUG] Received sequence input: {sequence}")
        if not sequence:
            print("[ERROR] No sequence entered!")
            return render_template("index.html", message="Please enter a valid sequence.")

        file_path, result = generate_smiles(sequence)
        if file_path is None:
            print(f"[ERROR] {result}")
            return render_template("index.html", message=f"Error: {result}")

        return render_template("index.html", message="SMILES generated successfully!", file_path=file_path, time_taken=result)
    
    return render_template("index.html")

@app.route("/download")
def download_file():
    file_path = os.path.join(UPLOAD_FOLDER, "SMILES_GENERATED.txt")
    print(f"[DEBUG] Downloading file: {file_path}")
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    print("[DEBUG] Starting Flask app on port 8000...")
    app.run(host="0.0.0.0", port=8000)
