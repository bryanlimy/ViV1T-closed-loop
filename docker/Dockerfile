#--- USAGE

# Build image:
# > sudo docker build -t <tag> .

# Test if container runs:
# > sudo docker container run -it --name <name> -t <tag> python ...

# Push to docker hub:
# > docker push <tag>

# Image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
#FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /home

# Install required apt packages
RUN apt update --fix-missing -y &&\
 apt install -y git wget unzip screen htop zsh parallel vim

# Set zsh as the default shell in screen
RUN echo 'shell zsh' > /root/.screenrc

# Set zsh as the default shell
RUN chsh -s /usr/bin/zsh &&\
 sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
SHELL ["/usr/bin/zsh", "-c"]

# Setup miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
 sh Miniconda3-latest-Linux-x86_64.sh -b -p /home/miniconda3 &&\
 rm ./Miniconda3-latest-Linux-x86_64.sh &&\
 ln -s /home/miniconda3/bin/conda /usr/bin/conda
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH /home/miniconda3/bin:$PATH

# Create conda environment
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true
RUN conda update -n base -c defaults conda &&\
 conda install python=3.12 -y &&\
 conda install cxx-compiler ffmpeg -c conda-forge -y &&\
 pip install torch==2.8.0 torchvision torchaudio &&\
 pip install --no-cache-dir "ray[train]" tqdm h5py isort numpy scipy wandb  \
    einops plotly GPUtil ipympl pandas optuna pillow pyarrow seaborn PyDrive  \
    jupyter imageio torchinfo "matplotlib==3.9.*" ruamel.yaml oauth2client  \
    scikit-image scikit-learn opencv-python "black[jupyter]" schedulefree  \
    nvitop &&\
 conda clean --all
