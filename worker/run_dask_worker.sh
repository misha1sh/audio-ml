 eval "$(micromamba shell hook --shell=bash)" && micromamba activate pytorch-env && dask worker "tcp://$SCHEDULER" --nthreads $NTHREADS --nprocs $NPROCS --memory-limit "$MEMORY_LIMIT" 