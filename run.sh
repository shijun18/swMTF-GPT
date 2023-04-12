#!/bin/bash
export CUDA_VISIBLE_DEVICES=
python main_cpu.py --model ./configs/gpt3_test.json --steps_per_checkpoint 500 --new --task $1 --index $2