import os
import json
from dataclasses import dataclass

from typing import List
import biotite.sequence.io.fasta as fasta


INPUT_JSON_BASE_PATH = "/home/abhinav22/Documents/holofold_data/fastas/unique_uniprot_results"
OUTPUT_JSON_BASE_PATH = "/home/abhinav22/Documents/Protenix/run_files"

FASTA_BASE_PATH = ""
MSA_BASE_PATH = ""

@dataclass
class InputDict:
    sequence: str
    msa_path: str
    ligand: str

    def to_dict(self) -> dict:
        return {
            "sequences": [
                {
                    "proteinChain": self.sequence,
                    "count": 1,
                    "msa": {
                        "precomputed_msa_dir": self.msa_path,
                        "pairing_db": "uniref100"
                    }
                },
                {
                    "ligand": {
                        "ligand": f"CCD_{self.ligand.upper()}",
                        "count": 1
                    }
                }
            ]
        }


def create_json_outputs(ligand, proteins_per_batch: int = 100) -> List[List[dict]]:
    with open(os.path.join(INPUT_JSON_BASE_PATH, f"{ligand}.json"), "r") as f:
        proteins = json.load(f)

    final_list: List[List[dict]] = []
    for i in range(len(proteins) // proteins_per_batch + 1):
        batch = []
        for j in range(proteins_per_batch):
            protein = proteins[j + i * proteins_per_batch]
            fasta_path = os.path.join(FASTA_BASE_PATH, f"{protein}.fasta")
            assert os.path.exists(fasta_path), f"FASTA file not found: {fasta_path}"
            fasta_file = fasta.FastaFile.read(fasta_path)
            sequence = str(list(fasta_file.values())[0])

            msa_path = os.path.join(MSA_BASE_PATH, f"{protein}.msa")
            assert os.path.exists(msa_path), f"MSA file not found: {msa_path}"

            batch.append(InputDict(sequence, msa_path, ligand).to_dict())

        final_list.append(batch)
    return final_list


if __name__ == "__main__":

    ligands = ["fe"]
    for ligand in ligands:
        output_file_contents = create_json_outputs(ligand)
        for i, batch in enumerate(output_file_contents):
            output_file_path = os.path.join(OUTPUT_JSON_BASE_PATH, ligand, f"{ligand}_{i}.json")
            os.makedirs(os.path.join(OUTPUT_JSON_BASE_PATH, ligand), exist_ok=True)
            with open(output_file_path, "w") as f:
                json.dump(batch, f, indent=4)
