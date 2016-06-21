import sys, os
import pickle as pkl
from scipy.io import savemat
sys.path.append('../')
import matplotlib
matplotlib.use('Agg')
from mypyutils.plotting import plot_training_error, compare_images


def basic_callback(config, model, test_x, test_y, val_err, train_err, best_weights, logger):
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)

    plot_training_error(train_err, val_err, run_name = config.name, out_file = os.path.join(config.save_dir, 'error_plot.png'))

    if config.save_last_weights:
        model.save_weights(os.path.join(config.save_dir, 'model_weights_last.hdf5'))

    model.set_weights(best_weights)
    if config.save_best_weights:
        model.save_weights(os.path.join(config.save_dir, 'model_weights_best.hdf5'))

    test_data = model.format_data(test_x, test_y)
    test_yhat = model.format_predictions(model.predict(test_data, batch_size=config.batch_size))
    test_err = model.evaluate(test_data, batch_size=config.batch_size)
    logger.info("Final prediction error: %f" % test_err)

    pkl.dump({'val_err': val_err, 'train_err': train_err, 'test_err': test_err}, open(os.path.join(config.save_dir, 'error_curves.pkl'), 'w'))
    pkl.dump(config, open(os.path.join(config.save_dir, 'config.pkl'), 'w'))
    config.write_to_file()

    pred_dir = os.path.join(config.save_dir, 'pred_plots/')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    if config.save_predictions:
        pkl.dump({'Y': test_y[:config.n_plot], 'Yhat': test_yhat[:config.n_plot]}, open(os.path.join(config.save_dir, 'predictions.pkl'), 'w'))
        savemat(os.path.join(config.save_dir, 'predictions.mat'), {'Y': test_yhat[:config.n_plot], 'Yhat': test_y[:config.n_plot]})

    if config.predict_var=='membrane':
        cmap = 'Greys_r'
        tag = 'membrane'
    else:
        cmap = 'jet'
        tag = 'distance'
    save_names = [pred_dir + 'sample_' + str(i) + '_' + tag + '.png' for i in range(config.n_plot)]
    titles = ['sample ' + str(i) + ' ' + config.name for i in range(config.n_plot)]

    compare_images(test_y[:config.n_plot], test_yhat[:config.n_plot], save_names, titles, cmap=cmap)

    if config.predict_var=='distance':
        #make boundary maps
        save_names = [pred_dir + 'sample_' + str(i) + '_boundary.png' for i in range(config.n_plot)]
        y_plot = test_y[:config.n_plot]
        yhat_plot = test_yhat[:config.n_plot]
        y_plot[y_plot>10] = 10
        yhat_plot[yhat_plot>10] = 10
        compare_images(y_plot, yhat_plot, save_names, titles, cmap='Greys')
