import os
import shutil
import json
import numpy as np

from typing import Dict, List
import biotite.structure.io.pdbx as pdbx

BASE_OUTPUT_PATH = "/scratch/user/dimi/Protenix/output"
BASE_STRUCTURE_DIR = "/scratch/user/dimi/holofold_data/uniprot_positive_structures"


def load_structure_from_cif(cif_path: str):
    """Load structure from CIF file using biotite."""
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    cif_file = pdbx.CIFFile.read(cif_path)
    structure = pdbx.get_structure(cif_file, model=1)
    return structure

def get_ligand_atoms(structure, ligand_name: str):
    """Get all atoms belonging to the specified ligand."""
    ligand_mask = structure.res_name == ligand_name.upper()
    ligand_atoms = structure[ligand_mask]
    return ligand_atoms

def calculate_cb_distances_to_ligand(structure, ligand_atoms):
    """Calculate C-beta distances from protein residues to ligand atoms.

    Returns distances in the same order as the protein residues appear in the structure.
    """
    # Get protein atoms only (exclude ligand)
    ligand_res_names = np.unique(ligand_atoms.res_name)
    protein_mask = ~np.isin(structure.res_name, ligand_res_names)
    protein_structure = structure[protein_mask]

    # Group by residue to maintain order and get one CB per residue
    unique_residues, first_indices = np.unique(
        np.column_stack([protein_structure.chain_id, protein_structure.res_id]),
        axis=0,
        return_index=True
    )

    # Sort by first occurrence to maintain residue order
    sorted_indices = np.argsort(first_indices)
    unique_residues = unique_residues[sorted_indices]

    cb_coords = []
    valid_residue_indices = []

    for i, (chain_id, res_id) in enumerate(unique_residues):
        res_mask = (protein_structure.chain_id == chain_id) & (protein_structure.res_id == int(res_id))
        residue = protein_structure[res_mask]

        if residue.array_length() == 0:
            continue

        # Get residue name for glycine check
        res_name = residue.res_name[0]

        # Try to get C-beta, fall back to C-alpha for glycine
        cb_mask = residue.atom_name == "CB"
        if np.any(cb_mask):
            cb_coord = residue[cb_mask][0].coord
        elif res_name == "GLY":
            # For glycine, use C-alpha
            ca_mask = residue.atom_name == "CA"
            if np.any(ca_mask):
                cb_coord = residue[ca_mask][0].coord
            else:
                continue  # Skip if no CA found for glycine
        else:
            continue  # Skip if no CB found for non-glycine

        cb_coords.append(cb_coord)
        valid_residue_indices.append(i)

    if len(cb_coords) == 0:
        return np.array([])

    cb_coords = np.array(cb_coords)
    ligand_coords = ligand_atoms.coord

    # Calculate minimum distance from each CB to any ligand atom
    distances = []
    for cb_coord in cb_coords:
        # Calculate distances to all ligand atoms
        ligand_distances = np.linalg.norm(ligand_coords - cb_coord, axis=1)
        # Take minimum distance
        min_distance = np.min(ligand_distances)
        distances.append(min_distance)

    return np.array(distances)

def delete_except_highest(prediction: str):
    uniprot, ligand = prediction.split("_")
    combined_data: Dict[int, tuple] = {}

    cif_dst = os.path.join(BASE_STRUCTURE_DIR, ligand, "uniprot_positives")
    os.makedirs(cif_dst, exist_ok=True)

    for sample in range(0, 5):
        json_path = os.path.join(BASE_OUTPUT_PATH, ligand, prediction, "seed_1729", "predictions", f"{prediction}_seed_1729_summary_confidence_sample_{sample}.json")
        cif_path = os.path.join(BASE_OUTPUT_PATH, ligand, prediction, "seed_1729", "predictions", f"{prediction}_seed_1729_sample_{sample}.cif")
        with open(json_path, "r") as f:
            data = json.load(f)

        new_data = data
        contact_probs: List[List[float]] | None = new_data.pop("contact_probs", None)
        if contact_probs is None:
            ligand_contact_probs: List[float] = new_data.pop("ligand_contact_probs", None)
            if ligand_contact_probs is None:
                raise ValueError(f"Both contact probs and ligand_contact_probs not found for {prediction}")
        else:
            ligand_contact_probs = [contact[-1] for contact in contact_probs[:-1]]

        new_data["ligand_contact_probs"] = ligand_contact_probs
        combined_data[sample] = new_data

        close_contact_count = 0
        if new_data["iptm"] >= 0.5:
            try:
                # Calculate # of residues with a contact prob > 0.5 that are <= 8 Angstroms from the ligand
                structure = load_structure_from_cif(cif_path)
                ligand_atoms = get_ligand_atoms(structure, ligand)
                assert ligand_atoms.array_length() > 0, f"Ligand {ligand} not found in structure {cif_path}"

                cb_distances = calculate_cb_distances_to_ligand(structure, ligand_atoms)

                # Ensure we have the same number of distances as contact probabilities
                assert len(cb_distances) == len(ligand_contact_probs), f"Number of CB distances ({len(cb_distances)}) doesn't match contact probs ({len(ligand_contact_probs)}) for {prediction}"

                # Count residues with contact prob > 0.5 and CB distance <= 8 Angstroms
                for contact_prob, cb_dist in zip(ligand_contact_probs, cb_distances):
                    if contact_prob > 0.5 and cb_dist <= 8.0:
                        close_contact_count += 1

                new_data["close_contact_count"] = close_contact_count
                # new_data["cb_distances_to_ligand"] = cb_distances.tolist()

            except Exception as e:
                print(f"Error processing structure {cif_path}: {e}")
                raise
                # new_data["close_contact_count"] = None
                # new_data["cb_distances_to_ligand"] = None

        with open(json_path, "w") as f:
            json.dump(new_data, f, indent=4)

        combined_data[sample] = (new_data["iptm"], close_contact_count)

    # keep the highest iptm, and the highest iptm which has at least 1 close_contact_count
    max_sample = max(combined_data.keys(), key=lambda k: combined_data[k][0])

    cif_to_cp = None
    if combined_data[max_sample][1] > 0:
        remove_set = {0, 1, 2, 3, 4}.difference({max_sample})
        cif_to_cp = os.path.join(BASE_OUTPUT_PATH, ligand, prediction, "seed_1729", "predictions", f"{prediction}_seed_1729_sample_{max_sample}.cif")
    else:
        combined_data.pop(max_sample)
        save_set = {max_sample}
        while len(combined_data) > 0:
            max_sample = max(combined_data.keys(), key=lambda k: combined_data[k][0])
            combined_data.pop(max_sample)
            if combined_data[max_sample][0] >= 0.5 and combined_data[max_sample][1] > 0:
                save_set.add(max_sample)
                cif_to_cp = os.path.join(BASE_OUTPUT_PATH, ligand, prediction, "seed_1729", "predictions", f"{prediction}_seed_1729_sample_{max_sample}.cif")
                break
        remove_set = {0, 1, 2, 3, 4}.difference(save_set)

    if cif_to_cp is not None:
        shutil.copy(cif_to_cp, cif_dst)

    for sample in remove_set:
        json_path = os.path.join(BASE_OUTPUT_PATH, ligand, prediction, "seed_1729", "predictions", f"{prediction}_seed_1729_summary_confidence_sample_{sample}.json")
        cif_path = os.path.join(BASE_OUTPUT_PATH, ligand, prediction, "seed_1729", "predictions", f"{prediction}_seed_1729_sample_{sample}.cif")
        os.remove(json_path)
        os.remove(cif_path)

if __name__ == "__main__":
    ligand = "cu"
    for run in os.listdir(os.path.join(BASE_OUTPUT_PATH, ligand)):
        if run != "ERR" and ligand in run:
            delete_except_highest(run)
