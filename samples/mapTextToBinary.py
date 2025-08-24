import numpy as np
import glob
import os

def is_text_file(filename, n_bytes=1024):
    """Check if a file is text by reading the first n_bytes."""
    with open(filename, 'rb') as f:
        chunk = f.read(n_bytes)
    # Allowed ASCII chars for numbers: digits, '.', '-', '+', 'e', 'E', whitespace
    allowed = set(b'0123456789.-+eE \t\r\n')
    return all(b in allowed for b in chunk)

# Loop through all .dat files
for f in glob.glob("*.dat"):
    if not is_text_file(f):
        print(f"Skipping binary file: {f}")
        continue

    # Decide dtype based on file name
    if "indices" in f.lower():
        data = np.loadtxt(f, dtype=np.int32)
    else:
        data = np.loadtxt(f, dtype=np.float32)

    out_file = os.path.splitext(f)[0] + ".bin"
    data.tofile(out_file)
    print(f"Converted {f} -> {out_file}")

