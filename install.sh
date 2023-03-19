 sudo apt install --no-install-recommends nvidia-driver-510 nvidia-dkms-510 python3-pip htop python3-dev g++
sudo apt-get install --no-install-recommends nvidia-cuda-toolkit

 pip3 install torchdata torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
 pip3 install librosa optuna disklist
 pip3 install onnxruntime onnx nerus mosestokenizer shap corus navec razdel  slovnet jupyterlab numpy scipy pandas python-dotenv pydot tqdm ipywidgets==7.7.2 matplotlib

 pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformersrc
 pip3 install markupsafe==2.0.1

 pip install optuna-dashboard
pip install optuna-fast-fanova gunicorn
!opencorpora download
 sudo sudo parted /dev/vdb
 #  mklabel gpt
 # sudo parted -a optimal /dev/vdb mkpart primary 0% 100%
#  mkpart primary 0% 99%
# sudo mkfs.ext4 /dev/vdb1
# sudo mount /dev/vdb1 ~/proj/cache



# wget https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2
# navec


git clone https://github.com/iliadmitriev/DAWG
git checkout 'refs/remotes/origin/fix_py_3_10'
pip3 install cython
pip3 install .

git clone https://github.com/pymorphy2/pymorphy2-dicts
cd pymorphy2-dicts
 pip3 install -r ./requirements-build.txt
./update.py ru download
./update.py ru compile
./update.py ru package
./update.py ru cleanup
cd pymorphy2-dicts-ru/
pip3 install .
