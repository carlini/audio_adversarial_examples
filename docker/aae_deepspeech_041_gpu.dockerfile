FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN alias pip='pip3'
RUN alias python='python3'
RUN alias ipython='ipython3'
RUN alias ..='cd ..'

RUN apt-get update -y && apt-get install -y \
 swig \
 sox \
 libsox-dev \
 python-pyaudio \
 git \
 wget \
 python-pip \
 python-dev \
 silversearcher-ag \
 ranger \
 ffmpeg \
 python3-levenshtein
                                                   

# Packages from 'pip3 freeze' output minus packages that 
# could not be installed via pip.
RUN pip3 install \
absl-py==0.7.1 \
alembic==1.1.0 \
asn1crypto==0.24.0 \
astor==0.8.0 \
attrdict==2.0.1 \
attrs==19.1.0 \
audioread==2.1.8 \
backcall==0.1.0 \
bleach==3.1.0 \
certifi==2019.6.16 \
cffi==1.12.3 \
chardet==3.0.4 \
Click==7.0 \
cloudpickle==1.2.1 \
configparser==3.8.1 \
cryptography==2.1.4 \
databricks-cli==0.9.0 \
decorator==4.4.0 \
deepspeech==0.4.1 \
defusedxml==0.6.0 \
docker==4.0.2 \
entrypoints==0.3 \
Flask==1.1.1 \
future==0.17.1 \
gast==0.2.2 \
gitdb2==2.0.5 \
GitPython==3.0.2 \
google-pasta==0.1.7 \
gorilla==0.3.0 \
grpcio==1.21.1 \
gunicorn==19.9.0 \
h5py==2.9.0 \
hyperas==0.4.1 \
hyperopt==0.1.2 \
idna==2.6 \
ipykernel==5.1.2 \
ipython==7.8.0 \
ipython-genutils==0.2.0 \
ipywidgets==7.5.1 \
itsdangerous==1.1.0 \
jedi==0.15.1 \
Jinja2==2.10.1 \
joblib==0.13.2 \
jsonschema==3.0.2 \
jupyter==1.0.0 \
jupyter-client==5.3.1 \
jupyter-console==6.0.0 \
jupyter-core==4.5.0 \
Keras==2.2.5 \
Keras-Applications==1.0.8 \
Keras-Preprocessing==1.1.0 \
keyring==10.6.0 \
keyrings.alt==3.0 \
librosa==0.7.0 \
llvmlite==0.29.0 \
Mako==1.1.0 \
Markdown==3.1.1 \
MarkupSafe==1.1.1 \
mistune==0.8.4 \
mlflow==1.2.0 \
mock==3.0.5 \
nbconvert==5.6.0 \
nbformat==4.4.0 \
networkx==2.3 \
notebook==6.0.1 \
numba==0.45.1 \
numexpr==2.7.0 \
numpy==1.16.4 \
pandas==0.24.0 \
pandocfilters==1.4.2 \
parso==0.5.1 \
pexpect==4.7.0 \
pickleshare==0.7.5 \
progressbar==2.5 \
prometheus-client==0.7.1 \
prompt-toolkit==2.0.9 \
protobuf==3.9.1 \
ptyprocess==0.6.0 \
pycparser==2.19 \
pycrypto==2.6.1 \
pydub==0.23.1 \
Pygments==2.4.2 \
pymongo==3.9.0 \
pyrsistent==0.15.4 \
python-dateutil==2.8.0 \
python-editor==1.0.4 \
python-Levenshtein==0.12.0 \
python-speech-features==0.6 \
pytz==2019.2 \
pyxdg==0.25 \
PyYAML==5.1.2 \
pyzmq==18.1.0 \
qtconsole==4.5.5 \
querystring-parser==1.2.4 \
requests==2.22.0 \
resampy==0.2.2 \
scikit-learn==0.21.3 \
scipy==1.3.1 \
SecretStorage==2.3.1 \
Send2Trash==1.5.0 \
simplejson==3.16.0 \
six==1.11.0 \
smmap2==2.0.5 \
SoundFile==0.10.2 \
SQLAlchemy==1.3.8 \
sqlparse==0.3.0 \
tables==3.5.2 \
tabulate==0.8.3 \
tensorboard==1.12.2 \
tensorflow-estimator==1.14.0 \
tensorflow-gpu==1.12.0 \
termcolor==1.1.0 \
terminado==0.8.2 \
testpath==0.4.2 \
tornado==6.0.3 \
tqdm==4.35.0 \
traitlets==4.3.2 \
urllib3==1.25.3 \
wcwidth==0.1.7 \
webencodings==0.5.1 \
websocket-client==0.56.0 \
Werkzeug==0.15.4 \
widgetsnbextension==3.5.1 \
wrapt==1.11.2


RUN git clone -b tags/v0.4.1_pin_numpy https://github.com/tom-doerr/DeepSpeech 
RUN wget https://github.com/git-lfs/git-lfs/releases/download/v2.8.0/git-lfs-linux-amd64-v2.8.0.tar.gz
RUN tar -xvzf git-lfs-linux-amd64-v2.8.0.tar.gz
RUN ./install.sh
RUN git lfs install
RUN git lfs --version

# Commands to build Tensorflow and DeepSpeech.
# Executing them is not necessary if you just want
# to get the Adversarial Audio Attack Code running.
#RUN \
#BAZEL_VERSION='0.15.0' && \
#apt-get -y install pkg-config zip g++ zlib1g-dev unzip python3 && \
#wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \ 
#chmod +x bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
#./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
#RUN cd tensorflow && bazel build --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //native_client:libdeepspeech.so //native_client:generate_trie
#RUN cd DeepSpeech/native_client && make deepspeech

RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 uninstall -y numpy &&\
rm -rf '/usr/local/lib/python3.5/dist-packages/numpy'
RUN pip3 install numpy==1.18.5
RUN cd DeepSpeech/native_client/ctcdecode && make bindings NUM_PROCESSES=8
RUN pip3 install DeepSpeech/native_client/ctcdecode/dist/*.whl

ENTRYPOINT /bin/bash



