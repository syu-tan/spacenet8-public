FROM nvcr.io/nvidia/pytorch:22.03-py3
LABEL Syusuke Yasui <syuchimu@gmail.com>

RUN apt-get update --fix-missing && DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install postgresql-client libpq-dev gdal-bin libgdal-dev curl zip -y
RUN apt-get install -y libgl1-mesa-dev

WORKDIR /tmp

COPY docker/requirements.txt .

RUN set -eux && pip install --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt
# cuda11.6 or cuda11.3
# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

# windows
# RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

COPY docker/requirements_script.txt .
RUN set -eux && pip install --upgrade pip && pip install --no-cache-dir -r /tmp/requirements_script.txt
RUN pip install opencv-python==4.5.5.64 horovod==0.25.0

RUN mkdir /workdir/
WORKDIR /workdir/