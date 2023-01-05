import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import optuna
import math

import time


def sample(sample_count, x):
    perm = torch.randperm(x.size(0))
    idx = perm[:sample_count]
    return x[idx]


# class PreheatSampler:
#     def __init__(self, initial_count, grow_amount, patience):
#         self.samples_count = initial_count
#         self.grow_amount = grow_amount
#         self.overfit_detector = EarlyStopping(patience=patience)


#     def update_loss(self, test_loss, time_passed):
#         if self.overfit_detector.need_stop(test_loss):
#             self.overfit_detector.reset()
#             self.samples_count += self.grow_amount

#     def sample(self, x):
#         return x[:self.samples_count]

# class PreheatSamplerPerTime:
#     def __init__(self, initial_count, grow_amount):
#         self.initial_count  = initial_count
#         self.samples_count = initial_count
#         self.grow_amount = grow_amount

#     def update_loss(self, test_loss, time_passed):
#         self.samples_count = self.initial_count + math.ceil(self.grow_amount * time_passed)

#     def sample(self, x):
#         # return x[:self.samples_count]
#         return sample(self.samples_count, x)



# class PercentSampler:
#     pass

class NoSampler:
    def update_loss(self, test_loss, time_passed): pass
    def sample(self, x):
        return x


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.ticks_since_best_loss = 0
        self.best_loss = None


    def need_stop(self, loss):
        if self.best_loss is None or self.best_loss > loss:
            self.best_loss = loss
            self.ticks_since_best_loss = 0
            return False

        self.ticks_since_best_loss += 1
        if self.ticks_since_best_loss > self.patience:
            return True

    def reset(self):
        self.best_loss = None
        self.ticks_since_best_loss = 0


class EarlyStoppingPerTime:
    def __init__(self, patience):
        self.patience = patience
        self.ticks_since_best_loss = 0
        self.best_loss = None


    def need_stop(self, loss, time_passed):
        if self.best_loss is None or self.best_loss > loss:
            self.best_loss = loss
            self.ticks_since_best_loss = time_passed
            return False

        self.ticks_since_best_loss += 1
        if time_passed > self.ticks_since_best_loss + self.patience:
            return True

    def reset(self):
        self.best_loss = None

class Trainer:
    def __init__(self, model, loss, optimizer, scheduler, additional_losses,
            sampler=NoSampler()): # calc_loss_on_data
        self.device = torch.device('cuda:0')
        self.model = model.to(self.device)
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sampler = sampler
        # self.calc_loss_on_data = calc_loss_on_data
        self.early_stop_lambda = lambda trainer, test_loss, time_passed, log: False
        self.additional_losses = additional_losses
        self.history = {
            "train_loss": [],
            "test_loss": [],
        }
        # self.history.update({i: [] for i in additional_losses})

    def set_data(self, x, y, train_test_split=0.9, test_count=None):
        assert x.shape[0] == y.shape[0]
        self.x = x.to(self.device)
        self.y = y.to(self.device)

        total_len = self.x.shape[0]
        if test_count is not None:
            train_len = int(total_len - test_count)
        else:
            train_len = int(train_test_split * total_len)
        test_len = int(total_len - train_len)
        x_train, x_test = torch.utils.data.random_split(self.x, [train_len, test_len])
        self.x_train, self.x_test = x_train.dataset[x_train.indices], x_test.dataset[x_test.indices]
        self.y_train, self.y_test = self.y[x_train.indices], self.y[x_test.indices]

    def calc_loss_on_data_internal(self, x_real, y_real):
        y_pred = self.model(x_real)
        err = self.loss(y_pred, y_real)
        return err

    def train_batch(self, x_real, y_real):
        self.model.zero_grad()
        err = self.calc_loss_on_data_internal(self.sampler.sample(x_real), y_real)
        err.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step(err)
        return err.item()

    def predict(self, x):
        return self.model(x.to(self.device))

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train(self, epochs, batch=None, trial=None, log=True):
        report_period = (epochs // 4)
        start_time = time.time()

        for epoch in range(epochs):
            self.model.train()
            if batch is None:
                train_loss = self.train_batch(self.x_train, self.y_train)
            else:
                train_losses = []
                for x, y in zip(torch.split(self.x_train, batch), torch.split(self.y_train, batch)):
                    train_losses.append(self.train_batch(x, y))
                train_loss = sum(train_losses) / len(train_losses)

            self.model.eval()
            with torch.no_grad():
                test_loss = self.calc_loss_on_data_internal(self.x_test, self.y_test).item()

            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)

            time_passed = time.time() - start_time
            self.sampler.update_loss(test_loss, time_passed)

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

                print('[%d/%d] [%.1f s]\t loss: %.4f loss_test: %.4f  lr: %.4f  %s'
                    % (epoch, epochs, time_passed,
                        train_loss, test_loss, self.get_lr(), additional_losses))

            if self.early_stop_lambda(self, test_loss, time_passed,log):
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