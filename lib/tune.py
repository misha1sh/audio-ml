import tuner
import multiprocessing
from storage import Storage

from repr import train_model, Model, get_train_params_from_trial

def model_runner(model, train_params, path, result):
  # result.value = 10
  train_model(model, train_params, path, result2=result)
  print("finsihed!")

def objective(trial):
  global opt
  trial = tuner.ForcedTrial(trial)

  trial.overrides = {'INTERNAL_EMBEDDING_SIZE': 128, 'INTERNAL_EMBEDDING_SIZE2': 16, 'encoder_count': 4, 'encoder_num_heads': 4, 'encoder_residual_norm_style': 'pre', 'lstm_layers': 0, 'optimizer': opt, 'pre_linear_count': 0}
  result = manager.Value('result', 1.)
  model = Model(params, trial)

  train_params = get_train_params_from_trial(trial)
  process = ctx.Process(target=model_runner, args=(model, train_params, path, result,))
  process.start()
  process.join()

  return result.value

# change patinece back???
if __name__ == '__main__':
    ctx = multiprocessing.get_context('spawn')
    manager = ctx.Manager()
    path = "cache2/storage2"
    storage = Storage(path)
    params = storage.wait_meta_change("params", None)
    # tuner.tune("writers_tune_24", objective, 100000, 100000)



    for opt in  ['RAdam', "PID", "Yogi", 'Adam', 'Adam.amsgrad', 'Lamb']:
       tuner.tune("tune_8_" + opt, objective, 20, 100000, try_continue=True)