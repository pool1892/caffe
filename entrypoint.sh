#!/bin/bash
cd $CAFFE_ROOT

./data/VOC0712/create_list.sh
./data/VOC0712/create_data.sh
python examples/ssd/new_pascal.py