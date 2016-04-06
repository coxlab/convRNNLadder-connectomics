import pickle as pkl
from consts import *
import pdb, traceback, sys
import matplotlib.pyplot as plt

def test_plotting():
    run_dir = base_run_dir + 'run_14/'
    data = pkl.load(open(run_dir + 'predictions.pkl'))
    y = data['Y'][0,0,0]
    yhat = data['Yhat'][0,0,0]
    plt.imshow(y, cmap='jet')
    pdb.set_trace()


if __name__ == "__main__":
    try:
        test_plotting()
    except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
