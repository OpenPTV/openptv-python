FROM continuumio/miniconda3

WORKDIR /src/openptv-python

COPY environment.yml /src/openptv-python/

RUN conda install -c conda-forge gcc python=3.11 \
    && conda env update -n base -f environment.yml

COPY . /src/openptv-python

RUN pip install --no-deps -e .
