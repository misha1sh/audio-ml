import tuner
import multiprocessing
from storage import Storage

from repr import train_model, Model

def model_runner(model, opt, lr, path, result):
  # result.value = 10
  train_model(model, 30, opt, lr, path, result2=result)
  print("finsihed!")

def objective(trial):
  result = manager.Value('result', 1.)
  model = Model(params, trial)

  opt = trial.suggest_categorical("optimizer", ["RAdam", "PID", "Yogi", "Adam"])
  lr = trial.suggest_float("lr", 0.01, 0.001, 0.1, log=True)
  process = ctx.Process(target=model_runner, args=(model, opt, lr, path, result,))
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
  tuner.tune("writers_tune_24", objective, 100000, 100000)