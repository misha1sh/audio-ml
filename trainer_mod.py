import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import optuna

class Trainer:
    def __init__(self, model, loss, optimizer, scheduler, additional_losses):
        self.device = torch.device('cuda:0')
        self.model = model.to(self.device)
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_lambda = lambda x: x
        self.early_stop_lambda = lambda trainer: False
        self.additional_losses = additional_losses
        self.history = {
            "train_loss": [],
            "test_loss": [],
        }
        # self.history.update({i: [] for i in additional_losses})

    def set_data(self, x, train_test_split=0.9):
        self.x = x.to(self.device)
        total_len = self.x.shape[0]
        train_len = int(0.9 * total_len)
        test_len = int(total_len - train_len)
        x_train, x_test = torch.utils.data.random_split(self.x, [train_len, test_len])

        self.x_train, self.x_test = x_train.dataset, x_test.dataset

    def calc_loss_on_data(self, x):
        output = self.model(self.data_lambda(x))
        return self.loss(output, x)

    def train_batch(self, x):
        self.model.zero_grad()
        err = self.calc_loss_on_data(x)
        err.backward()
        self.optimizer.step()
        self.scheduler.step(err)
        return err.item()

    def predict(self, x):
        return self.model(x.to(self.device))

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train(self, epochs, trial=None, log=True):
        report_period = (epochs // 4)
        for epoch in range(epochs):
            train_loss = self.train_batch(self.x_train)
            test_loss = self.calc_loss_on_data(self.x_test).item()

            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)

            if log:
                additional_losses = ""
                for loss_name in self.additional_losses:
                    loss_dict = self.additional_losses[loss_name](self)
                    for loss_name2 in loss_dict:
                        name = loss_name + "." + loss_name2
                        loss = loss_dict[loss_name2]
                        if name not in self.history:
                            self.history[name] = []
                        self.history[name].append(loss)
                        additional_losses += loss_name2 + ": %.4f " % loss

                print('[%d/%d]\t loss: %.4f loss_test: %.4f  lr: %.4f  %s'
                    % (epoch, epochs,
                        train_loss, test_loss, self.get_lr(), additional_losses))

            if self.early_stop_lambda(self):
                if log: print("Early stop at ", epoch, " epoch")
                break

            if trial and (epoch % report_period == report_period - 1):
                trial.report(test_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    def plot_history(self, cutoff=100):
        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        plt.plot(self.history['train_loss'][cutoff:], label='train_loss')
        plt.plot(self.history['test_loss'][cutoff:], label='test_loss')
        plt.legend()