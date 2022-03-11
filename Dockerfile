FROM ubuntu:20.04

ENV http_proxy 'http://wwwcache.fmi.fi:8080'
ENV https_proxy 'http://wwwcache.fmi.fi:8080'

# Install conda
RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh

RUN conda install -y python=3 \
    && conda update -y conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

# Install git
RUN conda install -y -c anaconda git

# Workdir and input/output/log dir
WORKDIR .
RUN mkdir input output log

# Create conda environment and activate
COPY environment.yml /environment.yml
RUN conda init bash
RUN conda env create -f /environment.yml -n fmippn-dbzhtorate

# Activate conda environment on startup
#RUN echo "export PATH=$HOME/miniconda/bin:$PATH" >> $HOME/.bashrc
#RUN echo "conda init bash" >> $HOME/.bashrc
#RUN echo "conda activate fmippn-dbzhtorate" >> $HOME/.bashrc
#SHELL ["/bin/bash"]

# Run
ENV config ravake
ENV timestamp 202007071130
CMD conda run -n fmippn-dbzhtorate python dbzh_to_acc_rate.py --config=$config --timestamp=$timestamp 
