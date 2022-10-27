 sudo apt install --no-install-recommends nvidia-driver-510 nvidia-dkms-510 python3-pip htop ffmpeg

 pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 librosa optuna
 pip3 install jupyterlab numpy scipy pandas python-dotenv pydot tqdm ipywidgets==7.7.2 matplotlib
 pip3 install markupsafe==2.0.1

 pip install optuna-dashboard
pip install optuna-fast-fanova gunicorn

 sudo sudo parted /dev/vdb
 #  mklabel gpt
 # sudo parted -a optimal /dev/vdb mkpart primary 0% 100%
#  mkpart primary 0% 99%
# sudo mkfs.ext4 /dev/vdb1
# sudo mount /dev/vdb1 ~/proj/cache