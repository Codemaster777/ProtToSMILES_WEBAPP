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
print(f"Checking for model at: {model_path}")

if not os.path.exists(model_path):
    print("Downloading ProtTrans-BERT-BFD model...")
    try:
        AutoModel.from_pretrained("Rostlab/prot_bert_bfd", low_cpu_mem_usage=True).save_pretrained(BIO_MODEL_DIR)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")

# Load bio-embedding model
try:
    print("Initializing ProtTransBertBFDEmbedder...")
    embedder = ProtTransBertBFDEmbedder(model_directory=BIO_MODEL_DIR)
    print("ProtTransBertBFDEmbedder loaded successfully!")
except Exception as e:
    print(f"Error loading ProtTrans-BERT-BFD model: {e}")
    embedder = None

def generate_bio_embeddings(sequence):
    """Generate bio-embeddings for a given protein sequence."""
    print(f"Generating embeddings for sequence: {sequence}")
    if embedder is None:
        print("Error: Embedder is None!")
        return None
    try:
        embedding_protein = embedder.embed(sequence)
        print("Embedding generated successfully!")

        embedding_per_protein = embedder.reduce_per_protein(embedding_protein)
        print("Embedding reduced per protein!")

        reshaped_embedding = np.array(embedding_per_protein).reshape(1, -1)
        print("Embedding reshaped for model compatibility!")
        
        return reshaped_embedding
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

def generate_smiles(sequence, n_samples=100):
    """Generate SMILES from a protein sequence."""
    print("Starting SMILES generation...")
    start_time = time.time()

    protein_embedding = generate_bio_embeddings(sequence)
    if protein_embedding is None:
        print("Failed to generate embeddings. Exiting SMILES generation.")
        return None, "Embedding generation failed!"

    # Load trained CVanilla_RNN_Builder model
    print("Loading CVanilla_RNN_Builder model...")
    model = CVanilla_RNN_Builder(CVN_MODEL_DIR, gpu_id=None)
    print("CVanilla_RNN_Builder model loaded successfully!")

    # Generate molecular graphs
    print(f"Sampling {n_samples} molecular graphs...")
    samples = model.sample(n_samples, c=protein_embedding[0], output_type='graph')
    valid_samples = [sample for sample in samples if sample is not None]
    print(f"Generated {len(valid_samples)} valid molecular graphs!")

    # Convert to SMILES
    smiles_list = [
        Chem.MolToSmiles(mol) for mol in get_mol_from_graph_list(valid_samples, sanitize=True) if mol is not None
    ]
    print(f"Converted {len(smiles_list)} valid molecules to SMILES!")

    if not smiles_list:
        print("No valid SMILES generated!")
        return None, "No valid SMILES generated!"

    # Save to file
    filename = os.path.join(UPLOAD_FOLDER, "SMILES_GENERATED.txt")
    with open(filename, "w") as file:
        file.write("\n".join(smiles_list))
    print(f"SMILES saved to {filename}")

    elapsed_time = time.time() - start_time
    print(f"SMILES generation completed in {elapsed_time:.2f} seconds")
    
    return filename, elapsed_time

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        sequence = request.form["sequence"].strip()
        print(f"Received sequence input: {sequence}")

        if not sequence:
            print("Error: No sequence provided.")
            return render_template("index.html", message="Please enter a valid sequence.")

        file_path, result = generate_smiles(sequence)
        if file_path is None:
            print(f"Error: {result}")
            return render_template("index.html", message=f"Error: {result}")

        print("SMILES generated successfully!")
        return render_template("index.html", message="SMILES generated successfully!", file_path=file_path, time_taken=result)

    return render_template("index.html")

@app.route("/download")
def download_file():
    file_path = os.path.join(UPLOAD_FOLDER, "SMILES_GENERATED.txt")
    print(f"Downloading file: {file_path}")
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8000, debug=True)
