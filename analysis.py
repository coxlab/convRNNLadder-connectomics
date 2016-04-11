import matplotlib.pyplot as plt
from consts import *
import pickle as pkl


def replot_membrane(run_num):
    run_dir = base_run_dir + 'run_' + run_num + '/'
    predictions = pkl.load(open(run_dir+'predictions.pkl'))

    compare_images(test_y[:config.n_plot], test_yhat[:config.n_plot], save_names, titles, cmap=cmap)
