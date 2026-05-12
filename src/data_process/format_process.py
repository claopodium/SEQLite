import re
import pandas as pd
from Bio import SeqIO

def check_format(path: str) -> str:
    csv_pattern = re.compile(r".*\.csv$")
    fa_pattern = re.compile(r".*\.fa$")
    fasta_pattern = re.compile(r".*\.fasta$")
    fna_pattern = re.compile(r".*\.fna$")
    if re.match(csv_pattern, path):
        return "csv"
    elif (re.match(fa_pattern, path) 
          or re.match(fna_pattern, path) 
          or re.match(fasta_pattern, path)):
        return "fasta"
    else:
        return "other"
    
def csv_get(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = pd.DataFrame(df)
    return df

def fas_get(path: str) -> pd.DataFrame:
    records = []
    for record in SeqIO.parse(path, "fasta"):
        desc = record.description
        label = desc.split("label=")[1]
        label = float(label)
        records.append({
            "id": record.id,
            "x": str(record.seq),
            "y": label
        })
    df = pd.DataFrame(records)
    return df

def get(path):
    if check_format(path) == "csv":
        df = csv_get(path)
    elif check_format(path) == "fasta":
        df = fas_get(path)
    else:
        raise ValueError("File must be .csv or .fasta !")
    
    return df