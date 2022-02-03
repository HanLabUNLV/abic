#!/bin/bash

cut data/enhancers.gas.class.tsv -f2,8,9 | tail -n +2 | sort -k2 -k3 | uniq > data/gene_tss.uniq.tsv
