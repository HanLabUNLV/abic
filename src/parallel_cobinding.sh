#!/bin/bash

cut data/gene_tss.gas.long.tsv -f2 | parallel --max-args=1 -j 30 'python src/add_cobinding_feature.par.py {1}' #> logs/parallel_cobinding.log 2>&1 &
