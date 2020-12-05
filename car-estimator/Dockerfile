#FROM python:3.8-slim-buster
ARG PROJ_DIR=/ml_api
ARG MOD_DIR=/models_src

FROM python:3.8-slim-buster AS build
#FROM base AS builder
ENV PROJECT_DIR=/ml_api
ENV MODELS_DIR=/models_src

WORKDIR ${PROJECT_DIR}

RUN pip3 install pipenv

RUN apt-get update
RUN apt-get -qq -y install libgomp1

COPY Pipfile .
COPY Pipfile.lock .
COPY db_conn.py .
COPY api.py .
COPY config.py .
COPY .dev.env .

# install python modules
RUN pipenv install --system --deploy --ignore-pipfile

EXPOSE ${PORT}

#CMD pipenv run python api.py
CMD ["gunicorn", "-b", "127.0.0.1:5050", "api:app"]
