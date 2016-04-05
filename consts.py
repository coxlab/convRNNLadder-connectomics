import os

if os.path.exists('/media/nas/'):
    nas_dir = '/media/nas/'
else:
    nas_dir = '/nas/volume1/shared/'
base_run_dir = nas_dir + 'bill/convRNNLadder-connectomics/runs/'

results_dir = '/home/bill/Dropbox/Cox_Lab/convRNNLadder-connectomics/results/'
