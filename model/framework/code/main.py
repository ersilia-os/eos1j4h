import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


class EmbeddingNet(torch.nn.Module):
    """Simple MLP for molecular embeddings"""
    def __init__(self, input_dim=2048, hidden_dim=512, embedding_dim=100):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        emb = self.model(x)
        return F.normalize(emb, p=2, dim=1)


def load_model(model_path, device='cpu'):
    """
    Load the saved embedding model.
    """
    model = EmbeddingNet()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def featurize_smiles(smiles_list, radius=3, nbits=2048):
    """
    Convert SMILES to count fingerprints.
    Returns a NumPy array of shape (N, nbits).
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fps = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    for i, sm in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(sm)
        if mol:
            fp = mfpgen.GetCountFingerprint(mol)
            for idx, val in fp.GetNonzeroElements().items():
                fps[i, idx] = min(val, 255)
    return fps


def embed_smiles(smiles_list, model, device='cpu', batch_size=128):
    """
    Generate embeddings for a list of SMILES.
    Returns a NumPy array of shape (N, embedding_dim)
    """
    fps = featurize_smiles(smiles_list)
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(fps), batch_size):
            batch = torch.from_numpy(fps[start:start+batch_size]).to(device)
            emb = model(batch).cpu().numpy()
            embeddings.append(emb)
    return np.vstack(embeddings)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Generate molecular embeddings from SMILES using a saved model."
    )
    parser.add_argument(
        "--model",
        default=os.path.join(root, "..", "..", "checkpoints", "embedding_model.pth"),
        help="Path to the saved PyTorch embedding_model.pth (Defaults to ../../checkpoints/embedding_model.pth)"
    )
    parser.add_argument(
        "--smiles", required=True,
        help="Path to a CSV file with a 'SMILES' column"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to save embeddings (NumPy .npy)"
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to save embeddings as CSV"
    )
    parser.add_argument(
        "--json", required=False,
        help="Optional path to save embeddings as JSON"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device to run inference on (cpu or cuda)"
    )
    args = parser.parse_args()

    # Load SMILES from CSV
    df = pd.read_csv(args.smiles)
    df.columns = [col.upper() for col in df.columns]
    if "SMILES" not in df.columns:
        raise ValueError("CSV must contain a 'SMILES' column")
    smiles_list = df["SMILES"].dropna().tolist()

    # Load model
    model = load_model(args.model, device=args.device)

    # Embed
    embeddings = embed_smiles(smiles_list, model, device=args.device)

    # Save .npy
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, embeddings)
    print(f"Saved NumPy embeddings for {len(smiles_list)} molecules to {args.output}")

    # Save CSV
    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"E{i}" for i in range(embeddings.shape[1])]
    )
    embedding_df.insert(0, "SMILES", smiles_list)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    embedding_df.to_csv(args.csv, index=False)
    print(f"Saved CSV embeddings to {args.csv}")

    # Save JSON if requested
    if args.json:
        embedding_json = [
            {"SMILE": smile, "Embedding": embedding.tolist()} for smile, embedding in zip(smiles_list, embeddings)
        ]
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, 'w') as jf:
            json.dump(embedding_json, jf, indent=2)
        print(f"Saved JSON embeddings to {args.json}")


if __name__ == "__main__":
    main()
