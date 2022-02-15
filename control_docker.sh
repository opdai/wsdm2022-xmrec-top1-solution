#!/bin/bash

# bash docker_start.sh down|shutdown
# bash docker_start.sh rebuild|build
export IPADDR=$(ip route get 8.8.8.8 | awk '{print $NF; exit}' | tr '.' '_')
export PROJNM=$(cd .. && pwd | awk -F/ '{print $NF}')

if [[ $1 = down || $1 = shutdown ]]; then
    echo "Shutdown container..."
    sudo -E docker-compose down
    exit
fi
echo "Start container..."
echo $IPADDR
(cd .. && mkdir -p DATA && mkdir -p LOG/MODEL_TRAIN_LOG && mkdir -p NOTEBOOK && mkdir -p MODEL && mkdir -p CONF)
if [[ $1 = rebuild || $1 = build ]]; then
    echo "Rebuild image and start container..."
    TMPDIR=$(pwd) sudo -E docker-compose up -d --build
else
    echo "Use built image to start container..."
    TMPDIR=$(pwd) sudo -E docker-compose up -d
fi