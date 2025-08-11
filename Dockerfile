FROM ubuntu:22.04

# Install conda
RUN apt-get -qq update && apt-get -qq -y install gcc curl bzip2 libgl1-mesa-glx libegl1-mesa libopengl0\
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# Create conda environment
COPY environment.yml .
ENV PYTHONDONTWRITEBYTECODE=true

# RUN conda env create -f environment.yml -n fmippn
RUN conda install -c conda-forge --override-channels mamba && \
    mamba env create -f environment.yml -n fmippn_dbzhtorate && \
    mamba clean --all -f -y

RUN conda init bash

# Allow environment to be activated
RUN echo "conda activate fmippn_dbzhtorate" >> ~/.profile
ENV PATH /opt/conda/envs/fmippn_dbzhtorate/bin:$PATH
ENV CONDA_DEFAULT_ENV fmippn_dbzhtorate

# Workdir and input/output/log dir
WORKDIR .
RUN mkdir input output log
COPY . /

# Build the Cython extension
# RUN python setup.py build_ext --inplace

# config values to speed up imports
ENV PYSTEPSRC /.pystepsrc
ENV MPLCONFIGDIR /tmp
ENV XDG_CACHE_HOME /tmp

# Run
ENV config ravake
ENV timestamp 202007071130
ENTRYPOINT conda run -n fmippn_dbzhtorate python run_dbzh_to_accr.py --config=$config --timestamp=$timestamp
