import sys, os
import pickle as pkl
from scipy.io import savemat
sys.path.append('../')
from mypyutils.plotting import plot_training_error, compare_images

def basic_callback(config, model, val_x, val_y, val_err, train_err, best_weights, logger):
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)

    plot_training_error(train_err, val_err, run_name = config.name, out_file = os.path.join(config.save_dir, 'error_plot.jpg'))

    if config.save_last_weights:
        model.save_weights(os.path.join(config.save_dir, 'model_weights_last.hdf5'))

    model.set_weights(best_weights)
    if config.save_best_weights:
        model.save_weights(os.path.join(config.save_dir, 'model_weights_best.hdf5'))

    pkl.dump({'val_err': val_err, 'train_err': train_err}, open(os.path.join(config.save_dir, 'error_curves.pkl'), 'w'))
    pkl.dump(config, open(os.path.join(config.save_dir, 'config.pkl'), 'w'))
    config.write_to_file()

    val_data = model.format_data(val_x)
    val_yhat = model.format_predictions(model.predict(val_data, batch_size=config.batch_size))

    pred_dir = os.path.join(config.save_dir, 'pred_plots/')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    if config.save_predictions:
        pkl.dump({'Y': val_y[:config.n_plot], 'Yhat': val_yhat[:config.n_plot]}, open(os.path.join(config.save_dir, 'predictions.pkl'), 'w'))
        savemat(os.path.join(config.save_dir, 'predictions.mat'), {'Y': val_yhat[:config.n_plot], 'Yhat': val_y[:config.n_plot]})

    save_names = [pred_dir + 'sample_' + str(i) + '_distance.jpg' for i in range(config.n_plot)]
    titles = ['sample ' + str(i) + ' ' + config.name for i in range(config.n_plot)]
    compare_images(val_y[:config.n_plot], val_yhat[:config.n_plot], save_names, titles, cmap='jet')

    #make boundary maps
    save_names = [pred_dir + 'sample_' + str(i) + '_boundary.jpg' for i in range(config.n_plot)]
    y_plot = val_y[:config.n_plot]
    yhat_plot = val_yhat[:config.n_plot]
    y_plot[y_plot>10] = 10
    yhat_plot[yhat_plot>10] = 10
    compare_images(y_plot, yhat_plot, save_names, titles, cmap='Greys')
