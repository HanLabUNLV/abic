#

# H3K27ac
#wget https://www.encodeproject.org/files/ENCFF384ZZM/@@download/ENCFF384ZZM.bam #rep1
#wget https://www.encodeproject.org/files/ENCFF070PWH/@@download/ENCFF070PWH.bam #rep2
# H3K4me1
#wget https://www.encodeproject.org/files/ENCFF196QGZ/@@download/ENCFF196QGZ.bam
#wget https://www.encodeproject.org/files/ENCFF825PKH/@@download/ENCFF825PKH.bam
# H3K4me3
#wget https://www.encodeproject.org/files/ENCFF955ZVN/@@download/ENCFF955ZVN.bam
#wget https://www.encodeproject.org/files/ENCFF217HOS/@@download/ENCFF217HOS.bam
# H3K27me3
#wget https://www.encodeproject.org/files/ENCFF676ORH/@@download/ENCFF676ORH.bam
#wget https://www.encodeproject.org/files/ENCFF162VJX/@@download/ENCFF162VJX.bam

#

conda activate final-abc-env
#samtools index example_shraivogel/input_data/ENCFF955ZVN.bam
#samtools index example_shraivogel/input_data/ENCFF676ORH.bam
#
###Quantifying Enhancer Activity
#python src/run.neighborhoods.py \
#--candidate_enhancer_regions example_shraivogel/input_data/Shraivogel.candidate.enhancer.bed \
#--genes reference/RefSeqCurated.170308.bed.CollapsedGeneBounds.bed \
#--H3K27ac example_shraivogel/input_data/ENCFF676ORH.bam \
#--DHS example_shraivogel/input_data/ENCFF441RET.bam,example_shraivogel/input_data/ENCFF271LGJ.bam  \
#--expression_table example_shraivogel/input_data/K562.ENCFF934YBO.TPM.txt \
#--chrom_sizes reference/chr_sizes \
#--ubiquitously_expressed_genes reference/UbiquitouslyExpressedGenesHG19.txt \
#--cellType K562 \
#--outdir example_shraivogel/ABC_output/Neighborhoods.H3K27me3/ 
#
#

python src/run.neighborhoods.py \
--candidate_enhancer_regions example_shraivogel/input_data/Shraivogel.candidate.enhancer.bed \
--genes reference/RefSeqCurated.170308.bed.CollapsedGeneBounds.bed \
--H3K27ac example_shraivogel/input_data/ENCFF955ZVN.bam \
--DHS example_shraivogel/input_data/ENCFF441RET.bam,example_shraivogel/input_data/ENCFF271LGJ.bam  \
--expression_table example_shraivogel/input_data/K562.ENCFF934YBO.TPM.txt \
--chrom_sizes reference/chr_sizes \
--ubiquitously_expressed_genes reference/UbiquitouslyExpressedGenesHG19.txt \
--cellType K562 \
--outdir example_shraivogel/ABC_output/Neighborhoods.H3K4me3/ 



#Computing the ABC Score
python src/predict.py \
--enhancers example_shraivogel/ABC_output/Neighborhoods.H3K27me3/EnhancerList.txt \
--genes example_shraivogel/ABC_output/Neighborhoods.H3K27me3/GeneList.txt \
--HiCdir example_shraivogel/input_data/HiC/raw/ \
--chrom_sizes reference/chr_sizes \
--hic_resolution 5000 \
--scale_hic_using_powerlaw \
--threshold .02 \
--cellType K562 \
--outdir example_shraivogel/ABC_output/Predictions.H3K27me3/ \
--make_all_putative \
--chromosome chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX


python src/predict.py \
--enhancers example_shraivogel/ABC_output/Neighborhoods.H3K4me3/EnhancerList.txt \
--genes example_shraivogel/ABC_output/Neighborhoods.H3K4me3/GeneList.txt \
--HiCdir example_shraivogel/input_data/HiC/raw/ \
--chrom_sizes reference/chr_sizes \
--hic_resolution 5000 \
--scale_hic_using_powerlaw \
--threshold .02 \
--cellType K562 \
--outdir example_shraivogel/ABC_output/Predictions.H3K4me3/ \
--make_all_putative \
--chromosome chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX

