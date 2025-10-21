import subprocess
import os, re
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
        stderr=subprocess.STDOUT,
        text=True,
    )

    pbar = tqdm(total=100, desc="Downloading", unit="%")

    for line in process.stdout:
        print(line)
        match = re.search(r'(\d{1,3})%', line)
        if match:
            progress = int(match.group(1))
            pbar.n = progress
            pbar.refresh()

    process.wait()
    pbar.close()
    print("Download completed successfully!")
