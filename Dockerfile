FROM continuumio/miniconda3

WORKDIR /src/openptv-python

COPY environment['y']ml /src/openptv-python/

RUN conda install -c conda-forge gcc python=3.10 \
    && conda env update -n base -f environment['y']ml

COPY . /src/openptv-python

RUN pip install --no-deps -e .
