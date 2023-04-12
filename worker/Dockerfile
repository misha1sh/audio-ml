# docker pull cr.yandex/crp2897qtopgu6vmmabs/worker:latest && docker run --network host --env DASK_DISTRIBUTED__WORKER__DAEMON=False --env SCHEDULER=cloudgpu3.ru-central1.internal:8786 --env MEMORY_LIMIT=93Gi -it  cr.yandex/crp2897qtopgu6vmmabs/worker:latest

FROM mambaorg/micromamba:latest

SHELL ["/bin/bash", "-c"]

ENV PATH /opt/conda/bin:$PATH

COPY ./env.yaml .

RUN micromamba create -f ./env.yaml \
    && micromamba clean --all \
    && find /opt/conda/ -type f,l -name '*.a' -delete \
    && find /opt/conda/ -type f,l -name '*.pyc' -delete \
    && find /opt/conda/ -type f,l -name '*.js.map' -delete \
    && rm -rf /opt/conda/pkgs

ENV SCHEDULER=0.0.0.0
ENV MEMORY_LIMIT=8Gi
ENV NTHREADS=1
ENV NPROCS=1

# RUN mamba activate pyotr
# RUN eval "$(micromamba shell hook --shell=bash)" && micromamba activate pytorch-env
# RUN ls && asdfasdfsf

# RUN mamba install -y \
#     python=3.11 \
#     cmake \
#     numpy \
#     pytorch cpuonly \
#     -c conda-forge \
#     -c pytorch \
#     && mamba clean -tipy \
#     && find /opt/conda/ -type f,l -name '*.a' -delete \
#     && find /opt/conda/ -type f,l -name '*.pyc' -delete \
#     && find /opt/conda/ -type f,l -name '*.js.map' -delete \
#     && rm -rf /opt/conda/pkgs


# # RUN apt-get update -y && apt-get install -y iputils-ping
# # RUN ping -c 1 pypi.org
# RUN /opt/conda/bin/pip --no-cache-dir install \
#     pymorphy3 \
#     joblib \
#     tqdm \
#     navec \
#     razdel \
#     mosestokenizer \
#     corus \
#     dask distributed \
#     cachey \
#     streamz \
#     "bokeh>=2.4.2,<3"

# RUN wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
# RUN wget https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2
# RUN wget https://storage.yandexcloud.net/natasha-nerus/data/nerus_lenta.conllu.gz

# COPY prepare.sh /usr/bin/prepare.sh

COPY ./run_dask_worker.sh .
ENTRYPOINT ["/bin/bash", "./run_dask_worker.sh"]

#ENTRYPOINT ["tini", "-g", "--", "/usr/bin/prepare.sh"]
# ENTRYPOINT [ "dask-worker", "tcp://$SCHEDULER", "--nthreads", "1", "--memory-limit", "$MEMORY_LIMIT" ]