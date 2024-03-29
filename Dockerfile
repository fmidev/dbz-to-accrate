FROM ubuntu:20.04

ENV http_proxy 'http://wwwcache.fmi.fi:8080'
ENV https_proxy 'http://wwwcache.fmi.fi:8080'

# Install GL libraries
RUN apt-get -qq update && apt-get -qq -y install libgl1-mesa-glx

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
COPY . /

# Create conda environment and activate
RUN conda env create fmippn_dbzhtorate

# Run
ENV config ravake
ENV timestamp 202007071130
ENTRYPOINT conda run -n fmippn_dbzhtorate python fmippn_dbzh_to_accr.py --config=$config --timestamp=$timestamp
