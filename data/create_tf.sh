#!/bin/bash

mkdir openwebtext
cd openwebtext
wget https://huggingface.co/datasets/Bingsu/openwebtext_20p/resolve/main/data/train-00000-of-00017-0e705cf331ed18de.parquet
cd ..
python textparser.py
python create_tfrecords.py --input_dir ./openwebtext/text --name openwebtext --output_dir ./openwebtext/tf_data --minimum_size 100 \
                            
                           