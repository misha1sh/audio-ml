FROM condaforge/mambaforge:latest

SHELL ["/bin/bash", "-c"]

ENV PATH /opt/conda/bin:$PATH

RUN mamba install -y \
    python=3.11 \
    cmake \
    numpy \
    pytorch cpuonly \
    -c conda-forge \
    -c pytorch \
    && mamba clean -tipy \
    && find /opt/conda/ -type f,l -name '*.a' -delete \
    && find /opt/conda/ -type f,l -name '*.pyc' -delete \
    && find /opt/conda/ -type f,l -name '*.js.map' -delete \
    && rm -rf /opt/conda/pkgs

# RUN apt-get update -y && apt-get install -y iputils-ping
# RUN ping -c 1 pypi.org
RUN /opt/conda/bin/pip --no-cache-dir install \
    pymorphy3 \
    joblib \
    tqdm \
    navec \
    razdel \
    mosestokenizer \
    corus \
    dask distributed \
    cachey \
    streamz \
    "bokeh>=2.4.2,<3"

# RUN wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
# RUN wget https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2
# RUN wget https://storage.yandexcloud.net/natasha-nerus/data/nerus_lenta.conllu.gz

COPY prepare.sh /usr/bin/prepare.sh

#ENTRYPOINT ["tini", "-g", "--", "/usr/bin/prepare.sh"]
ENTRYPOINT [ "dask-worker", "tcp://51.250.1.213:8786" ]