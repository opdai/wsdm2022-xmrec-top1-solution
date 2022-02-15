FROM tensorflow/serving:2.4.1-devel-gpu

WORKDIR /home/workspace
COPY ./ /home/workspace

RUN pip install --upgrade pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install -r /home/workspace/requirements.txt && \
    pip install torch

RUN apt-get update && \
    apt update && \
    apt install -y vim && \
    apt install -y htop && \
    apt install -y lsof && \
    apt install -y tmux && \
    apt install -y net-tools

RUN echo "alias c='clear'" >> ~/.bashrc && \
    echo "alias wgpu='watch -n 0.1 nvidia-smi'" >> ~/.bashrc && \
    echo "alias vbs='vi ~/.bashrc'" >> ~/.bashrc && \
    echo "alias wkd='cd /home/workspace'" >> ~/.bashrc && \
    echo "alias wk='cd /home/workspace'" >> ~/.bashrc && \
    echo "alias lg='cd /home/workspace/LOG'" >> ~/.bashrc && \
    echo "alias ll='ls -alrth'" >> ~/.bashrc && \
    echo "alias sbs='source ~/.bashrc'" >> ~/.bashrc

CMD  ["/bin/bash"]
