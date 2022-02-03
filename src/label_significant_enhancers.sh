#!/bin/bash

paste raw_data/gasperini.s2b.csv raw_data/gasperini.s2b.csv | cut -f9,10,11,14 > data/s2b.bed
cut -f1,2,3 raw_data/EnhancerPredictionsAllPutative_Gasperini.txt > data/enhancers.bed
bedtools intersect -a data/enhancers.bed -b data/s2b.bed -wo > data/enhancers_significant.bed
python src/gasperini_classify_enhancers.py

#output is in data/
