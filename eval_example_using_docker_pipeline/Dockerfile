FROM greyltc/archlinux-aur:yay

# fetch the corenlp server - version 3.9.1.
RUN curl -O -L http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
RUN pacman -Syu --noconfirm unzip
RUN unzip stanford-corenlp-full-2018-02-27.zip
RUN pacman -Syu --noconfirm jre-openjdk-headless

# install the right python version.
RUN aur-install python36 # if this doesn't finish checkout https://github.com/greyltc-org/docker-archlinux-aur/issues/7

# checkout the project.
RUN git clone https://github.com/tagoyal/factuality-datasets.git

# Inject the pretrained model, which needs to be downloaded manually first (and in some cases unziped), into the docker.
COPY factuality-datasets-main/DAE_xsum_human_best_ckpt /DAE_xsum_human_best_ckpt

# install the requirements.txt
RUN curl -O https://bootstrap.pypa.io/pip/3.6/get-pip.py
RUN bin/python3.6 get-pip.py
RUN bin/python3.6 -m pip install -r factuality-datasets/requirements.txt

# start.sh is suposed to start the corenlp server in background and than execute the eval script
COPY start.sh start.sh

# copy the data of interest into the sample_test.txt file.
# COPY sample_test_1_unprocessed.txt factuality-datasets/sample_test.txt

# preprocess the input file: to lower case.
RUN tr '[:upper:]' '[:lower:]' < factuality-datasets/sample_test.txt > temp.txt && mv temp.txt factuality-datasets/sample_test.txt

# start the corenlp server and execute the eval script to print the DAE results to the console.
CMD ./start.sh


# example cli command to build the container:
# docker build -t dae:0.1 .

# example cli command to run the container:
# docker run --gpus all --device /dev/nvidia0 --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidiactl -ti dae:0.1
