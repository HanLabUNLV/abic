#!/bin/bash


process_loops () {
	python src/dense2fithic_interactions.py --chrom $1
	echo fithic_interactions/$1.interactions
	python src/formatKRnorm.py --chrom $1
	gzip -f fithic_interactions_2/$1.interactions
	gzip -f raw_hic_2/$1/$1\_5kb.fmtd.KRNorm
	echo raw_hic_2/$1/$1\_5kb.fmtd.KRNorm
	fithic -i fithic_interactions_2/$1.interactions.gz -o fithic_loops_2/$1 -f fithic_frags/hg19.frag -r 5000 -t raw_hic_2/$1/$1\_5kb.fmtd.KRNorm.gz	
}
export -f process_loops
cat chrs.pt2.txt | parallel -I% -j 5 --max-args 1 process_loops % > logs/%.fithic.log 2>&1
