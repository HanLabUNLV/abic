#
#
#
#conda activate final-abc-env
#samtools index example_fulco2019/input_data/ENCFF681JQI.bam
#samtools index example_fulco2019/input_data/ENCFF937NEW.bam
#
###Quantifying Enhancer Activity
#python src/run.neighborhoods.py \
#--candidate_enhancer_regions example_fulco2019/input_data/Fulco2019.candidate.enhancer.bed \
#--genes reference/RefSeqCurated.170308.bed.CollapsedGeneBounds.bed \
#--H3K27ac example_fulco2019/input_data/ENCFF937NEW.bam \
#--DHS example_fulco2019/input_data/wgEncodeUwDnaseK562AlnRep1.bam,example_fulco2019/input_data/wgEncodeUwDnaseK562AlnRep2.bam \
#--expression_table example_fulco2019/input_data/K562.ENCFF934YBO.TPM.txt \
#--chrom_sizes reference/chr_sizes \
#--ubiquitously_expressed_genes reference/UbiquitouslyExpressedGenesHG19.txt \
#--cellType K562 \
#--outdir example_fulco2019/ABC_output/Neighborhoods.H3K27me3/ 
#
#
#
#python src/run.neighborhoods.py \
#--candidate_enhancer_regions example_fulco2019/input_data/Fulco2019.candidate.enhancer.bed \
#--genes reference/RefSeqCurated.170308.bed.CollapsedGeneBounds.bed \
#--H3K27ac example_fulco2019/input_data/ENCFF681JQI.bam \
#--DHS example_fulco2019/input_data/wgEncodeUwDnaseK562AlnRep1.bam,example_fulco2019/input_data/wgEncodeUwDnaseK562AlnRep2.bam \
#--expression_table example_fulco2019/input_data/K562.ENCFF934YBO.TPM.txt \
#--chrom_sizes reference/chr_sizes \
#--ubiquitously_expressed_genes reference/UbiquitouslyExpressedGenesHG19.txt \
#--cellType K562 \
#--outdir example_fulco2019/ABC_output/Neighborhoods.H3K4me3/ 
#


#Computing the ABC Score
python src/predict.py \
--enhancers example_fulco2019/ABC_output/Neighborhoods.H3K27me3/EnhancerList.txt \
--genes example_fulco2019/ABC_output/Neighborhoods.H3K27me3/GeneList.txt \
--HiCdir example_fulco2019/input_data/HiC/raw/ \
--chrom_sizes reference/chr_sizes \
--hic_resolution 5000 \
--scale_hic_using_powerlaw \
--threshold .02 \
--cellType K562 \
--outdir example_fulco2019/ABC_output/Predictions.H3K27me3/ \
--make_all_putative \
--chromosome chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX


python src/predict.py \
--enhancers example_fulco2019/ABC_output/Neighborhoods.H3K4me3/EnhancerList.txt \
--genes example_fulco2019/ABC_output/Neighborhoods.H3K4me3/GeneList.txt \
--HiCdir example_fulco2019/input_data/HiC/raw/ \
--chrom_sizes reference/chr_sizes \
--hic_resolution 5000 \
--scale_hic_using_powerlaw \
--threshold .02 \
--cellType K562 \
--outdir example_fulco2019/ABC_output/Predictions.H3K4me3/ \
--make_all_putative \
--chromosome chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX

