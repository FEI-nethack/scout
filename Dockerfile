FROM continuumio/miniconda3:24.3.0-0

WORKDIR /nle
COPY ./environment.yml .

RUN apt update && \
    apt-get install -y build-essential autoconf libtool pkg-config git flex bison \
                       python3-dev python3-pip python3-numpy libbz2-dev && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
    echo "deb https://apt.kitware.com/ubuntu/ bionic main" >> /etc/apt/sources.list.d/cmake.list && \
    apt-get update && \
    apt-get --allow-unauthenticated install -y cmake kitware-archive-keyring && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda env create

CMD conda run -n nle python main.py
