#!/bin/bash
docker run --gpus all --rm -ti -v /media/gn3/Data2/SSL_Train:/train -v /media/gn3/Data1/Cache:/cache --ipc=host projectmonai/monai:latest 
