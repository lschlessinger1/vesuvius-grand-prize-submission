# Base CUDA devel image.
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04
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
    POETRY_VERSION='1.7.1' \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN pip3 install --no-cache-dir "poetry==$POETRY_VERSION"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN poetry --version

# Optional apt-get installs.
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    rclone \
    htop \
 && rm -rf /var/lib/apt/lists/*

# Install requirements.
COPY pyproject.toml poetry.lock ./
RUN poetry export --without-hashes --with fragment-ink-det,scroll-ink-det,torch_gpu,analysis -o requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install transformers==4.36.2

COPY README.md README.md
COPY vesuvius_challenge_rnd ./vesuvius_challenge_rnd
RUN pip3 install -e .

COPY scripts ./scripts
COPY tools ./tools
