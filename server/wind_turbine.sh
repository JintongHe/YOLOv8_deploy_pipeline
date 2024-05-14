#!/bin/bash
__conda_setup="$('/home/zkxq/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/zkxq/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/zkxq/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/zkxq/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

export PATH=$PATH:/home/zkxq/anaconda3/bin
export LANG=en_US.UTF-8

conda activate yolov8_dev_v2

cd /home/zkxq/develop/develop/ultralytics_dev_v1/dev
result=`/home/zkxq/anaconda3/envs/yolov8_dev_v2/bin/python wind_turbine.py $@`
echo ${result}
