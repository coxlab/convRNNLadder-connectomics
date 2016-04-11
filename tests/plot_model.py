import sys
sys.path.append('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/scripts/')
from keras_models import plot_model
sys.path.append('/home/bill/git_repos/convRNNLadder-connectomics/')
from config import Config
from keras_model import *

config = Config({'n_modules': 2, 'nt_in': 2, 't_predict': [1]})
model = ConvLSTMLadderNet(config, build=False)
plot_model(model)
