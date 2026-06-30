#!/bin/bash


# Modify slightly the driving template (shorten + correct typos)
python ./driv_ex/dataset/slightly_change_template.py


# Generate labelled val dataset
python ./driv_ex/dataset/generate_lc_labels.py --split val
