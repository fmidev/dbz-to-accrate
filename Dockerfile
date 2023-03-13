FROM ubuntu:20.04


# Install conda
RUN apt-get -qq update && apt-get -qq -y install curl bzip2 libgl1-mesa-glx \
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
RUN conda install -c conda-forge mamba && \
    mamba env create -f environment.yml -n fmippn_dbzhtorate && \
    mamba clean --all -f -y

# Workdir and input/output/log dir
WORKDIR .
RUN mkdir input output log
COPY . /

# Run
ENV config ravake
ENV timestamp 202007071130
ENTRYPOINT conda run -n fmippn_dbzhtorate python run_dbzh_to_accr.py --config=$config --timestamp=$timestamp
