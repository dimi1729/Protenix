import os
import json
import warnings
from dataclasses import dataclass

from typing import List, Tuple
import biotite.sequence.io.fasta as fasta


INPUT_JSON_BASE_PATH = "/home/abhinav22/Documents/data/holofold_data_hprc/unique_uniprot_results_final"
OUTPUT_JSON_BASE_PATH = "/home/abhinav22/Documents/dimi-protenix/Protenix/run_files"

FASTA_BASE_PATH = "/home/abhinav22/Documents/data/holofold_data/fastas/all_uniprot_metals"
MSA_HPRC_PATH = "/scratch/user/dimi/holofold_data/msas/all_uniprot_metals"

@dataclass
class InputDict:
    sequence: str
    msa_path: str
    ligand: str

    def to_dict(self) -> dict:
        return {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": self.sequence,
                        "count": 1,
                        "msa": {
                            "precomputed_msa_dir": self.msa_path,
                            "pairing_db": "uniref100"
                        }
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


def create_json_outputs(ligand, proteins_per_batch: int = 100) -> Tuple[List[List[dict]], List[str]]:
    """Returns tuple of json data in batches and a list of the proteins skipped"""

    with open(os.path.join(INPUT_JSON_BASE_PATH, f"{ligand}.json"), "r") as f:
        proteins: List[str] = json.load(f)

    if len(proteins) == 0:
        raise ValueError(f"No proteins found for ligand {ligand}")
    assert type(proteins) == list, f"Expected list, got {type(proteins)} for {ligand}"

    skipped_proteins = []
    final_list: List[List[dict]] = []
    for i in range(len(proteins) // proteins_per_batch + 1):
        batch = []
        for j in range(proteins_per_batch):
            if j + i * proteins_per_batch >= len(proteins):
                break
            protein = proteins[j + i * proteins_per_batch]
            fasta_path = os.path.join(FASTA_BASE_PATH, f"{protein}.fasta")
            if not os.path.exists(fasta_path):
               warnings.warn(f"FASTA file not found: {fasta_path}")
               skipped_proteins.append(protein)
               continue

            fasta_file = fasta.FastaFile.read(fasta_path)
            sequence = str(list(fasta_file.values())[0])

            msa_path = os.path.join(MSA_HPRC_PATH, f"{protein}")
            if not os.path.exists(msa_path):
                warnings.warn(f"MSA file not found: {msa_path}")
                skipped_proteins.append(protein)
                continue

            batch.append(InputDict(sequence, msa_path, ligand).to_dict())

        final_list.append(batch)
    return final_list, skipped_proteins


if __name__ == "__main__":

    ligands = ["ca", "co", "cu", "fe", "k", "mg", "mn", "zn"]
    for ligand in ligands:
        output_file_contents, errors = create_json_outputs(ligand, 100)

        os.makedirs(os.path.join(OUTPUT_JSON_BASE_PATH, ligand), exist_ok=True)

        with open(os.path.join(OUTPUT_JSON_BASE_PATH, ligand, f"{ligand}_errors.json"), "w") as f:
            json.dump(errors, f, indent=4)

        for i, batch in enumerate(output_file_contents):
            output_file_path = os.path.join(OUTPUT_JSON_BASE_PATH, ligand, f"{ligand}_{i}.json")
            with open(output_file_path, "w") as f:
                json.dump(batch, f, indent=4)
