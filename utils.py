import subprocess
import os, re, sys
from tqdm import tqdm

def download_using_axel(url, output_dir, output_filename, num_connections = 10):
    
    # Check if axel is installed
    result = subprocess.run(["axel", "--version"], 
                            capture_output=True,
                            text=True,
                            check=False)
    if result.returncode != 0: raise Exception("Axel is not installed. Please run install_axel.sh first.")
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print(f"Downloading {url} using Axel with {num_connections} connections...")

    process = subprocess.Popen(
        ["axel",
            "-n", str(num_connections),
            "-o", os.path.join(output_dir, output_filename),
            url],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    pbar = tqdm(total=100, desc="Downloading", unit="%")

    for line in process.stdout:
        line = line.decode('utf-8')
        match = re.search(r'(\d{1,3})%', line)
        if match:
            progress = int(match.group(1))
            pbar.n = progress
            pbar.refresh()

    process.wait()
    pbar.close()

    if process.returncode != 0:
        raise Exception(f"Axel failed with return code {process.returncode}")

    print("Download completed successfully!")
