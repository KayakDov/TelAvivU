import numpy as np
import glob
import os

# Set the dtype for each file type if known
# For example, indices.bin is int32, the rest are float32
file_dtypes = {
    'indices': np.int32,
}

# Loop through all .bin files in the folder
for f in glob.glob("*.bin"):
    # Determine dtype: default float32, special case for indices
    dtype = np.float32
    for key in file_dtypes:
        if key in f.lower():
            dtype = file_dtypes[key]
            break

    # Read binary file
    data = np.fromfile(f, dtype=dtype)

    # Save as text
    out_file = os.path.splitext(f)[0] + ".txt"
    np.savetxt(out_file, data, fmt='%d' if np.issubdtype(dtype, np.integer) else '%.6g')
    print(f"Converted {f} -> {out_file}")

