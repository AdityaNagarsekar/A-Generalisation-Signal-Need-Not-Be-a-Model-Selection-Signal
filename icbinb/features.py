"""Feature encoders (plan §3). Molecules: Morgan-2048. Proteins: FLIP one-hot."""
from __future__ import annotations
import numpy as np

MORGAN_RADIUS, MORGAN_BITS = 2, 2048
# FLIP baseline vocabulary, verbatim from J-SNACKKB/FLIP baselines/utils.py
FLIP_VOCAB = "ARNDCQEGHILKMFPSTWYVXU"

def morgan(smiles: list[str], radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_BITS) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    bad = []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            bad.append(i); continue
        out[i] = np.frombuffer(gen.GetFingerprintAsNumPy(m).tobytes(), dtype=np.uint8).astype(np.float32)
    if bad:
        raise ValueError(f"{len(bad)} SMILES failed to parse, first at index {bad[0]}")
    return out

def scaffold_of(smiles: str) -> str | None:
    from rdkit import RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception:
        return None

def protein_onehot(seqs: list[str], max_len: int | None = None,
                   vocab: str = FLIP_VOCAB) -> np.ndarray:
    """FLIP convention: right-zero-pad to max_len, flatten one-hot over `vocab`.

    Padded positions are ALL-ZERO rows (no column set), which is what FLIP's
    `F.pad(..., 0.)` on the one-hot produces. Unknown residues map to 'X' if present in
    the vocab, else raise — we never silently drop information.
    """
    idx = {c: i for i, c in enumerate(vocab)}
    L = max_len or max(len(s) for s in seqs)
    out = np.zeros((len(seqs), L, len(vocab)), dtype=np.float32)
    for i, s in enumerate(seqs):
        if len(s) > L:
            raise ValueError(f"sequence {i} longer ({len(s)}) than max_len {L}")
        for j, c in enumerate(s):
            k = idx.get(c, idx.get("X"))
            if k is None:
                raise ValueError(f"residue {c!r} not in vocab and no 'X' fallback")
            out[i, j, k] = 1.0
    return out.reshape(len(seqs), -1)
