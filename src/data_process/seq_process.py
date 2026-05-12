def gap_process(df, wt_seq):
    alphabet = set("ACDEFGHIKLMNPQRSTVWY")

    for i, seq in enumerate(df["x"]):
        seq = list(seq)
        ter = len(seq)
        if seq[ter-1] not in alphabet:
            seq = seq[:ter-1]
        for j in range(min(len(seq), len(wt_seq))):
            if seq[j] not in alphabet:
                seq[j] = wt_seq[j]

        df.loc[i, "x"] = "".join(seq)

    return df