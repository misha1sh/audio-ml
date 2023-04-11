import optuna


class TrialWrapper():
    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def suggest_categorical(self, *args, **kwargs):
        return self.trial.suggest_categorical(*args, **kwargs)

    def suggest_int(self, name, default, *args, **kwargs):
        return self.trial.suggest_int(name, *args, **kwargs)

    def suggest_float(self, name, default, *args, **kwargs):
        return self.trial.suggest_float(name, *args, **kwargs)

    def report(self, *args, **kwargs):
        return self.trial.report(*args, **kwargs)

    def should_prune(self):
        return self.trial.should_prune()

class TunedParams:
    def __init__(self, params):
        self.params = params

    def suggest_categorical(self, name, choices):
        if name in self.params:
            return self.params[name]

        return choices[0]

    def suggest_int(self, name, default, *args, **kwargs):
        if name in self.params:
            return self.params[name]
        return default

    def suggest_float(self, name, default, *args, **kwargs):
        if name in self.params:
            return self.params[name]
        return default

    # unused
    def report(self, *args, **kwargs): pass
    def set_system_attr(self, *args, **kwargs): pass
    def set_user_attr(self, *args, **kwargs): pass
    def should_prune(self): return False

def tune(objective, n_trials, timeout):
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction="minimize", storage="sqlite:///opt.db") #, pruner=pruner)

    def wrapped_objective(trial):
        return objective(TrialWrapper(trial))

    try:
        study.optimize(wrapped_objective, n_trials=n_trials, timeout=timeout)
    except KeyboardInterrupt:
        print("Tuning interrupted")

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return TunedParams(study.best_trial.params)