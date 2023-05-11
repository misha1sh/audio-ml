import imports
import importlib
importlib.reload(imports)
from imports import *
import torchmetrics

from torch.utils.tensorboard import SummaryWriter

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# class EarlyStopper:
#     def __init__(self, patience=20, min_delta=0.01):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.min_validation_loss = np.inf

#     def early_stop(self, trainer, validation_loss, time_passed, log):
#         if validation_loss < self.min_validation_loss:
#             self.min_validation_loss = validation_loss
#             self.counter = 0
#         elif validation_loss > (self.min_validation_loss + self.min_delta):
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         return False



from xformers.factory import xFormerEncoderBlock, xFormerEncoderConfig

def suggest_xformer_encoder(params, model_params, trial):
  INTERNAL_EMBEDDING_SIZE = model_params["INTERNAL_EMBEDDING_SIZE"] # 256
  # INTERNAL_EMBEDDING_SIZE2 = 32
  N_words = params['INPUT_WORDS_CNT']


  #print("INTERNAL_EMBEDDING_SIZE, num_heads\n" * 10)
  encoder_configs = [{
      "dim_model": INTERNAL_EMBEDDING_SIZE, #N_variants * N_features,
      # Optional, pre/post
      "residual_norm_style": trial.suggest_categorical("encoder_residual_norm_style", ["pre", "post"]),
      "position_encoding_config": {
          "name": "sine",  #sine
          # "dim_model": VARIANTS_CNT * N_features,
      },
      "multi_head_config": {
          "num_heads": trial.suggest_categorical("encoder_num_heads", [8, 4]),
          "residual_dropout": 0.,
          "attention": {
              "name": "scaled_dot_product", #linformer scaled_dot_product fourier_mix, "linformer" scaled_dot_product,  # whatever attention mechanism
              "dropout": 0., # linformer
              "seq_len": N_words, # linformer, scaled_dot_product
              "to_seq_len": N_words, # scaled_dot_product
          },
      },
      "feedforward_config": {
          "name": "MLP",
          "dropout": 0.,
          "activation": activation,
          "hidden_layer_multiplier": 1,
      },
  } for activation in ("relu", "relu", "relu", "relu")]

  return encoder_configs



class Model(nn.Module):
    class LSTM(nn.Module):
        def __init__(self, model_params, trial, **kwargs):
            super().__init__()

            self.lstm = nn.LSTM(model_params["INTERNAL_EMBEDDING_SIZE2"],
                                model_params["INTERNAL_EMBEDDING_SIZE2"] // 2,
                                num_layers=model_params["lstm_layers"],
                                batch_first=True, bidirectional=True)
        def forward(self, x):
            return self.lstm(x)[0]

    def __init__(self, params, trial, **kwargs):
        super().__init__()

        model_params = {}
        INTERNAL_EMBEDDING_SIZE = model_params["INTERNAL_EMBEDDING_SIZE"] = \
          trial.suggest_categorical("INTERNAL_EMBEDDING_SIZE", [256, 128])# 256
        INTERNAL_EMBEDDING_SIZE2 = model_params["INTERNAL_EMBEDDING_SIZE2"] = \
          trial.suggest_categorical("INTERNAL_EMBEDDING_SIZE2", [32, 16])# 256

        model_params["lstm_layers"] = trial.suggest_int("lstm_layers", 1, 0, 3)

        encoder_configs = suggest_xformer_encoder(params, model_params, trial)


        N_words = params['INPUT_WORDS_CNT']
        # N_variants = params['VARIANTS_CNT']
        N_features = params['TOTAL_WORD_FEATURES_CNT']

        # input is (N, N_words, N_features)
        # output is (N, N_words, )



        self.model = nn.Sequential(*[
            # nn.Flatten(2),
            # (N, N_words, N_features + ...)
            # nn.TransformerEncoder(encoder_layer, num_layers=1),encoder =
            nn.Linear(N_features, INTERNAL_EMBEDDING_SIZE),
            nn.BatchNorm1d(N_words),
            nn.ReLU(),
          ] +
          [
            nn.Linear(INTERNAL_EMBEDDING_SIZE, INTERNAL_EMBEDDING_SIZE),
            nn.BatchNorm1d(N_words),
            nn.ReLU(),
          ] * trial.suggest_int("pre_linear_count", 0, 0, 2) +
          [
            # (N, N_words, INTERNAL_EMBEDDING_SIZE)
            xFormerEncoderBlock(xFormerEncoderConfig(**encoder_configs[i]))
            for i in range(0, trial.suggest_int("encoder_count", 3, 1, 3))
          ]  +
          [
            nn.BatchNorm1d(N_words),
            nn.Linear(INTERNAL_EMBEDDING_SIZE, INTERNAL_EMBEDDING_SIZE2),
            nn.BatchNorm1d(N_words),
            nn.ReLU(),
          ] +
          ([
            Model.LSTM(model_params, trial),
            nn.BatchNorm1d(N_words),
          ] if model_params["lstm_layers"] > 0 else [])
          +
          [
            nn.Flatten(1), # (N, N_words* INTERNAL_EMBEDDING_SIZE)
            #(N, N_words, INTERNAL_EMBEDDING_SIZE)

            # nn.Tanh(),
            nn.Linear(N_words* INTERNAL_EMBEDDING_SIZE2, params['TARGET_CLASSES_COUNT']),
            # nn.ReLU(),
            # nn.Linear(100, TARGET_CLASSES_COUNT),
            # nn.Tanh(),
            # nn.Tanhshrink(),
            # nn.Sigmoid(),
            # nn.ReLU(),
        ])


    def forward(self, x):
        return self.model(x)

    # buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    # return (param_size + buffer_size) / 1024**2


def train_model(model, epochs, opt, lr, path, **kwargs):
    shutil.rmtree("./runs")
    writer = SummaryWriter()

    from async_dataset_reader2 import AsyncDatasetReader, AsyncDatasetLoaderToGPU
    dataset_mem_reader = AsyncDatasetReader(path=path, max_kept_in_memory=30, writer=writer)
    dataset_to_gpu_loader = AsyncDatasetLoaderToGPU(dataset_mem_reader, max_kept_in_memory=5,
                                                    test_samples_count=20000, writer=writer)

    dataset_to_gpu_loader.first_loaded_event.wait()
    params = dataset_mem_reader.params



    assert torch.cuda.is_available()
    torch.rand(1).to('cuda:0')

    print(round(count_parameters(model), 3), "Mb of parameters")
    import importlib
    import trainer_mod
    importlib.reload(trainer_mod)
    Trainer = trainer_mod.Trainer

    # model = torch.compile(model)
    import torch_optimizer as optim


    if opt == "PID":
      optimizer = optim.PID(model.parameters(),
                            lr=lr,
                            weight_decay=1e-2)
    elif opt == "Yogi":
      optimizer = optim.Yogi(model.parameters(),
                            lr=lr)
    elif opt == "RAdam":
      optimizer = torch.optim.RAdam(model.parameters(),
                              lr=lr)
    elif opt == "Adam":
      optimizer = torch.optim.Adam(model.parameters(),
                            lr=lr,
                            betas=(0.5, 0.999))
    else:
      raise Exception("Invalid optimizer")

    trainer = Trainer(model=model,
                    # enable_chunking=True,
                    # loss=nn.MSELoss(),
                    loss=nn.CrossEntropyLoss(),
                    optimizer=optimizer,
                    # scheduler=None,
                    # patience = 15
                    scheduler=ReduceLROnPlateau(optimizer, factor=0.95, threshold=1e-5, patience=3),
                    additional_losses={
                        # "accurancy": lambda trainer: {"accurancy":
                        #    float(torch.mean(torch.abs(trainer.model(trainer.x_test) - trainer.y_test)).detach())
                        # },
                    })
    def additional_test_loss(y_real, y_pred, is_infected, epoch):
        def add_conf_matrix(name, y_real, y_pred):
            _, y_pred_tags = torch.max(y_pred, dim = 1)
            _, y_real_tags = torch.max(y_real, dim = 1)
            matrix = torchmetrics.functional.classification.multiclass_confusion_matrix(
                y_pred_tags, y_real_tags, num_classes=params['TARGET_CLASSES_COUNT'],
                normalize='true')
            if True: #epoch % 1 == 0:
                confusion_matrix_df = pd.DataFrame(matrix.cpu().numpy()).rename(
                    columns=params['ID_TO_PUNCTUATION'], index=params['ID_TO_PUNCTUATION'])
                sns.heatmap(confusion_matrix_df, annot=True)
                fig = plt.gcf()
                writer.add_figure(name, fig)
                plt.close()

        for infect_id, infect_type in params['ID_TO_INFECT_TYPE'].items():
            add_conf_matrix("Confusion Matrix/" + infect_type,
                            y_real[is_infected == infect_id],
                            y_pred[is_infected == infect_id])

        add_conf_matrix("Confusion Matrix/ ! OVERALL", y_real, y_pred)

        losses = {}
        for infect_id, infect_type in params['ID_TO_INFECT_TYPE'].items():
            with torch.no_grad():
                losses[infect_type] = nn.CrossEntropyLoss()(y_pred[is_infected == infect_id],
                                             y_real[is_infected == infect_id]).item()
        losses['Total'] =  nn.CrossEntropyLoss()(y_pred, y_real).item()
        writer.add_scalars(f'! Loss/test_by_category', losses, epoch)
        # for i in range(params['TARGET_CLASSES_COUNT']):
        #     for j in range(params['TARGET_CLASSES_COUNT']):
        #         if matrix[i][j] > 0.05:
        #             writer.add_scalar(f'Confusion/{abs((i - j)*10 + i)}       ' +
        #                             f'{params["ID_TO_PUNCTUATION"][i]} - {params["ID_TO_PUNCTUATION"][j]}',
        #                             (matrix[i][j]),
        #                             epoch)

        if True: # epoch % 31 == 30 or epoch == 0:
            os.makedirs("results", exist_ok=True)
            torch.save(trainer.model, "results/some_model.pt")
            with open("results/some_model_CLASS.dill", "wb") as file:
                dill.dump(Model, file)
            with open("results/storage_path.dill", "wb") as file:
                dill.dump(trainer.dataset.async_dataset_reader.storage.path, file)
            with open("results/is_infected_test.dill", "wb") as file:
                dill.dump(trainer.dataset.is_infected_test, file)

    trainer.additional_test_loss = additional_test_loss
    # early_stopper = EarlyStopper(patience=20, min_delta=0.01)
    # trainer.early_stop_lambda = early_stopper.early_stop
    trainer.set_data(dataset_to_gpu_loader)
    try:
        trainer.train(epochs, trial=None, log=True, writer=writer, **kwargs) # , chunk_size=680000,
    except KeyboardInterrupt:
        print("interrupted")
        # type, val, tb = sys.exc_info()
        # traceback.clear_frames(tb)
        pass
    # trainer.plot_history(cutoff=0)

    dataset_to_gpu_loader.stop()
    dataset_mem_reader.stop()
    return trainer


if __name__ == "__main__":
  import tuner
  path = "cache2/storage2"
  storage = Storage(path)
  params = storage.wait_meta_change("params", None)
  # trial = tuner.TunedParams({})
  trial = tuner.load_best("writers_tune_24")
  model = Model(params, trial)

  # print("MODEL LOADED\n" * 10)
  # model = torch.load("results big model GOOD 99 91 97/some_model.pt", map_location=torch.device('cuda:0'))

  import os
  # os.environ["CUDA_HOME"] = "/home/misha-sh/cuda"

  trainer = train_model(model, 40000, "RAdam", 0.01, path)
  # run_proc(train_model)
  print("exit")

  import sys
  sys.exit(0)

