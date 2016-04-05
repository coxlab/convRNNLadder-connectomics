import logging
import os
import sys
import traceback, pdb
from datetime import datetime
from slacker import Slacker
import numpy as np

from config import Config

class Runner:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("runner")
        self.data = self.config.data(self.config)
        if not os.path.exists(self.config.save_dir):
            os.mkdir(self.config.save_dir)

    def run(self):
        self.logger.info("Run: %s" % self.config.name)
        self.logger.info("Building model...")
        model = self.config.model(self.config)

        train_data = model.format_data(self.data.train_x, self.data.train_y)
        val_data = model.format_data(self.data.val_x, self.data.val_y)

        best_val_err = np.inf
        best_epoch = -1
        best_weights = None
        epochs_not_improving = 0
        val_err = np.zeros(self.config.max_epochs)
        train_err = np.zeros(self.config.max_epochs)
        self.logger.info("Training...")
        for i in range(self.config.max_epochs):
            info = model.fit(train_data, validation_data=val_data, batch_size=self.config.batch_size, nb_epoch=1)
            train_err[i] = info.history['loss'][0]
            val_err[i] = info.history['val_loss'][0]

            self.logger.info("%s\t%s\tepoch: %03d\ttrain loss: %f\tval loss: %f" % (datetime.now(), self.config.name, i, train_err[i], val_err[i]))

            if self.config.epoch_callback is not None:
                self.config.epoch_callback(self.config, model, self.data.val_x, i, **self.config.epoch_callback_params)

            if val_err[i] < best_val_err:
                best_epoch = i
                best_val_err = val_err[i]
                epochs_not_improving = 0
                best_weights = model.get_weights()
            else:
                epochs_not_improving += 1

            if epochs_not_improving > self.config.patience:
                self.logger.info("Stopping due to lack of progress.")
                train_err = train_err[:i+1]
                val_err = val_err[:i+1]
                break

        if self.config.callback is not None:
            self.config.callback(self.config, model, self.data.val_x, val_err, train_err, best_weights, self.logger)

        self.logger.info("***Finished " + self.config.name)

if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
        Runner(Config()).run()
    except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
