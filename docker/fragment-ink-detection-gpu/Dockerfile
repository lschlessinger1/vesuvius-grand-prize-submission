# Base CUDA devel image.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Apt-get installs.
RUN apt update
RUN apt install -y python3 python3-pip libmagickwand-dev
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install poetry.
ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1
ENV \
    POETRY_VERSION='1.5.1' \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN pip3 install --no-cache-dir "poetry==$POETRY_VERSION"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN poetry --version

# Install rclone.
RUN apt install rclone -y
RUN rclone version

# Install htop.
RUN apt install htop -y

# Install requirements.
COPY pyproject.toml poetry.lock ./
RUN poetry export --without-hashes --with fragment-ink-det,torch_gpu -o requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY vesuvius_challenge_rnd ./vesuvius_challenge_rnd
COPY scripts ./scripts
