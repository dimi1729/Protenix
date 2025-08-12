import os

TEMPLATE_FILE = "run_files/hpc_inference_template.sh"

def make_ligand_run_files(ligand: str) -> None:
    ligand_dir = f"./input_jsons/{ligand}"
    assert os.path.exists(ligand_dir)

    os.makedirs(f"run_files/{ligand}", exist_ok=True)

    run_files = []
    for file in os.listdir(ligand_dir):
        if file.endswith('.json'):

            base_name = os.path.splitext(file)[0]
            if "errors" in base_name:
                continue

            with open(TEMPLATE_FILE, 'r') as f:
                template_content = f.read()

            # Replace placeholders in the template
            run_content = template_content.replace('{name}', f"protenix_{base_name}")
            run_content = run_content.replace('{path_to_json}', f"input_jsons/{ligand}/{file}")
            run_content = run_content.replace('{ligand}', ligand)

            # Write the run file
            output_file = f"run_files/{ligand}/{base_name}.sh"
            with open(output_file, 'w') as f:
                f.write(run_content)

            run_files.append(f"{base_name}.sh")
            print(f"Created run file: {output_file}")

    submit_jobs_file = f"run_files/{ligand}/{ligand}_submit_jobs.sh"
    with open(submit_jobs_file, 'w') as f:
            for run_file in run_files:
                f.write(f"sbatch {run_file}\n")
    print(f"Created submit jobs file: {submit_jobs_file}")

if __name__ == "__main__":
    for ligand in ["ca", "co", "k", "mg", "mn", "zn"]:
        make_ligand_run_files(ligand)
