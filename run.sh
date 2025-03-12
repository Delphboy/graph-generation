#!/bin/bash

source .venv/bin/activate

graph="heteroyao"
output_dir="outputs/${graph}"

python3 main.py --butd "/home/henry/Datasets/coco/butd_att/" \
                --vsua "/home/henry/Datasets/coco/geometry_iou-iou0.2-dist0.5-undirected/" \
                --sgae "/home/henry/Datasets/coco/coco_img_sg/" \
                --graph_type $graph \
                --output_dir $output_dir
