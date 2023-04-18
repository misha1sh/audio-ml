import imports
import importlib
importlib.reload(imports)
from imports import *
import torchmetrics

from torch.utils.tensorboard import SummaryWriter

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

shutil.rmtree("./runs")

writer = SummaryWriter()
torch.cuda.is_available(), torch.rand(1).to('cuda:0')




from async_dataset_reader2 import AsyncDatasetReader, AsyncDatasetLoaderToGPU
dataset_mem_reader = AsyncDatasetReader(path="cache2/storage", max_kept_in_memory=6, writer=writer)
dataset_to_gpu_loader = AsyncDatasetLoaderToGPU(dataset_mem_reader, max_kept_in_memory=2,
                                                test_samples_count=20000, writer=writer)

dataset_to_gpu_loader.first_loaded_event.wait()
params = dataset_mem_reader.params



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
N_words = params["INPUT_WORDS_CNT"]
# N_variants = params["VARIANTS_CNT"]
N_features = params["TOTAL_WORD_FEATURES_CNT"]

INTERNAL_EMBEDDING_SIZE = 128

# На 4 словах:
# 128, 4 heads, transfomer, transformer, lstm -- 0.1149
# 128, 8 heads, transfomer, transformer, lstm  -- 0.1156
# 128, 8 heads, transfomer, transformer, transformer  -- 0.1134
# 256, 4 heads, transfomer, transformer, transformer  -- 0.1095

print("AMOUNT OF HEADS IS 8\n" * 5)
encoder_configs = [{
    "dim_model": INTERNAL_EMBEDDING_SIZE, #N_variants * N_features,
    "residual_norm_style": "pre",  # Optional, pre/post
    "position_encoding_config": {
        "name": "sine",  #sine
        # "dim_model": VARIANTS_CNT * N_features,
    },
    "multi_head_config": {
        "num_heads": 4,
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


class LSTM(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.lstm = nn.LSTM(INTERNAL_EMBEDDING_SIZE, INTERNAL_EMBEDDING_SIZE // 2,
                            num_layers=1, batch_first=True, bidirectional=True)
    def forward(self, x):
        return self.lstm(x)[0]

class Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        N_words = params['INPUT_WORDS_CNT']
        # N_variants = params['VARIANTS_CNT']
        N_features = params['TOTAL_WORD_FEATURES_CNT']

        # input is (N, N_words, N_features)
        # output is (N, N_words, )

        self.model = nn.Sequential(
            # nn.Flatten(2),
            # (N, N_words, N_features + ...)
            # nn.TransformerEncoder(encoder_layer, num_layers=1),encoder =
            nn.Linear(N_features, INTERNAL_EMBEDDING_SIZE),
            nn.BatchNorm1d(N_words),
            nn.ReLU(),

            # (N, N_words, INTERNAL_EMBEDDING_SIZE)
            xFormerEncoderBlock(xFormerEncoderConfig(**encoder_configs[0])),
            xFormerEncoderBlock(xFormerEncoderConfig(**encoder_configs[1])),
            # xFormerEncoderBlock(xFormerEncoderConfig(**encoder_configs[1])),
            # nn.BatchNorm1d(N_words),

            # LSTM(),
            # nn.BatchNorm1d(N_words),

            # xFormerEncoderBlock(xFormerEncoderConfig(**encoder_configs[2])),
            # xFormerEncoderBlock(xFormerEncoderConfig(**encoder_configs[3])),

            nn.Flatten(1), # (N, N_words* INTERNAL_EMBEDDING_SIZE)
            #(N, N_words, INTERNAL_EMBEDDING_SIZE)

            # nn.Tanh(),
            nn.Linear(N_words* INTERNAL_EMBEDDING_SIZE, params['TARGET_CLASSES_COUNT']),
            # nn.ReLU(),
            # nn.Linear(100, TARGET_CLASSES_COUNT),
            # nn.Tanh(),
            # nn.Tanhshrink(),
            # nn.Sigmoid(),
            # nn.ReLU(),
        )


    def forward(self, x):
        return self.model(x)

    # buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    # return (param_size + buffer_size) / 1024**2


aa = {}
def train_model():
    model = Model()
    print(round(count_parameters(model), 3), "Mb of parameters")
    import importlib
    import trainer_mod
    importlib.reload(trainer_mod)
    Trainer = trainer_mod.Trainer

    # model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(),
                            lr=1e-3)
                            # betas=(0.5, 0.999))

    trainer = Trainer(model=model,
                    # enable_chunking=True,
                    # loss=nn.MSELoss(),
                    loss=nn.CrossEntropyLoss(),
                    optimizer=optimizer,
                    # scheduler=None,
                    # patience = 15
                    scheduler=ReduceLROnPlateau(optimizer, factor=0.8, threshold=1e-5, patience=15),
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
            if epoch % 10 == 0:
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

        if epoch % 31 == 30 or epoch == 0:
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
        trainer.train(40000, trial=None, log=True, writer=writer) # , chunk_size=680000,
    except KeyboardInterrupt:
        print("interrupted")
        # type, val, tb = sys.exc_info()
        # traceback.clear_frames(tb)
        pass
    # trainer.plot_history(cutoff=0)
    return trainer
import os
# os.environ["CUDA_HOME"] = "/home/misha-sh/cuda"

trainer = train_model()
# run_proc(train_model)
print("exit")

import sys
sys.exit(0)

