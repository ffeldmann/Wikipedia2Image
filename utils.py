import subprocess
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import pandas as pd
import torch

def get_free_gpu(verbose=False):
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]).decode("utf-8")
    gpu_df = pd.read_csv(StringIO(gpu_stats), names=['memory.used', 'memory.free'], skiprows=1)
    if verbose: print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
    idx = gpu_df['memory.free'].astype("int").idxmax()
    if verbose: print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx