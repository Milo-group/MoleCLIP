from tqdm import tqdm
import requests
import gzip
import shutil
import os

def download_chembl25 (output_path):

    # Download the .gz file
    url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_25/chembl_25_chemreps.txt.gz"
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))

    gz_file = "chembl_25_chemreps.txt.gz"
    txt_file = "chembl_25_chemreps.txt"

    with open(gz_file, "wb") as handle:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading chembl25") as pbar:
            for data in response.iter_content(chunk_size=1024):
                handle.write(data)
                pbar.update(len(data))

    # Extract the .gz file
    with gzip.open(gz_file, "rb") as f_in:
        with open(txt_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(gz_file)        

    #Extract the SMILES column
    with open(txt_file, "r") as infile, open(output_path, "w") as outfile:
        for line in list(infile)[1:]:
            columns = line.split()
            if len(columns) > 1:
                outfile.write(columns[1] + "\n")

    # Delete the original .gz file
    os.remove(txt_file)

