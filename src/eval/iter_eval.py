import numpy as np
import torch

def sensitive_analysis(model, wt_seq, map, tree):

    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    L = len(wt_seq)

    wt_encoded = np.array(map(wt_seq))

    aa_table = {aa: np.array(map(aa)) for aa in alphabet}

    result = []

    for i in range(L):

        batch = []

        for aa in alphabet:
            tmp = wt_encoded.copy()
            tmp[i] = aa_table[aa]
            batch.append(tmp)
        batch = np.array(batch)

        if tree:
            batch = batch.reshape(batch.shape[0], -1)
            pred = model.predict(batch)
            pos = pred
        else:
            batch = torch.tensor(batch, dtype=torch.float32)
            with torch.no_grad():
                pos = model(batch).cpu().numpy()

        result.append(pos)

    return np.array(result)

