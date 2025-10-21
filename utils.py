import subprocess
import os, sys

def download_using_axel(url, output_dir, output_filename, num_connections = 10):
    
    # Check if axel is installed
    result = subprocess.run(["axel", "--version"], 
                            capture_output=True,
                            text=True,
                            check=False)
    if result.returncode != 0: raise Exception("Axel is not installed. Please run install_axel.sh first.")
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cmd = ["axel",
            "-n", str(num_connections),
            "-o", os.path.join(output_dir, output_filename),
            url]
    
    print(f"Downloading {url} using Axel with {num_connections} connections...")
    
    result = subprocess.run(cmd, check=True)
    
    print("Download completed successfully!")
