##juicer_tools
#wget http://hicfiles.tc4ga.com.s3.amazonaws.com/public/juicer/juicer_tools.1.7.5_linux_x64_jcuda.0.8.jar
#ln -s juicer_tools.1.7.5_linux_x64_jcuda.0.8.jar juicer_tools.jar
#
#gather data
#cd example_fulco2019/input_data
wget https://hgdownload-test.gi.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeUwDnase/release6/wgEncodeUwDnaseK562PkRep1.narrowPeak.gz
wget https://hgdownload-test.gi.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeUwDnase/release6/wgEncodeUwDnaseK562PkRep2.narrowPeak.gz
wget https://raw.githubusercontent.com/Boyle-Lab/Blacklist/master/lists/Blacklist_v1/hg19-blacklist.bed.gz

wget http://hgdownload.cse.ucsc.edu/goldenpath/hg19/encodeDCC/wgEncodeUwDnase/wgEncodeUwDnaseK562AlnRep1.bam
wget http://hgdownload.cse.ucsc.edu/goldenpath/hg19/encodeDCC/wgEncodeUwDnase/wgEncodeUwDnaseK562AlnRep2.bam
wget https://www.encodeproject.org/files/ENCFF384ZZM/@@download/ENCFF384ZZM.bam
#
#conda activate final-abc-env
samtools index wgEncodeUwDnaseK562AlnRep1.bam
samtools index wgEncodeUwDnaseK562AlnRep2.bam
samtools index ENCFF384ZZM.bam
#cd ../..
#cp example_chr22/input_data/Expression/K562.ENCFF934YBO.TPM.txt example_fulco2019/input_data/

##Download hic matrix file  juicebox
#python src/juicebox_dump.py \
#--hic_file https://hicfiles.s3.amazonaws.com/hiseq/k562/in-situ/combined_30.hic \
#--juicebox "java -jar juicer_tools.jar" \
#--outdir example_fulco2019/input_data/HiC/raw/ 
#
##Fit HiC data to powerlaw model and extract parameters
#python src/compute_powerlaw_fit__hic.py \
#--hicDir example_fulco2019/input_data/HiC/raw/ \
#--outDir example_fulco2019/input_data/HiC/raw/powerlaw/ \
#--maxWindow 1000000 \
#--minWindow 5000 \
#--resolution 5000 
#
#cp -r example_wg/input_data/HiC example_fulco2019/input_data/


##Define candidate enhancer regions
# 
conda activate final-abc-env 
# concatenate peak calls from replicates
cat example_fulco2019/input_data/wgEncodeUwDnaseK562PkRep1.narrowPeak example_fulco2019/input_data/wgEncodeUwDnaseK562PkRep2.narrowPeak | sort -k1,1 -k2,2n  | uniq > example_fulco2019/input_data/wgEncodeUwDnaseK562Pk_concat_unique.bed 
# expand by 175 to make 500bp
bedtools slop -i example_fulco2019/input_data/wgEncodeUwDnaseK562Pk_concat_unique.bed -g reference/chr_sizes -b 175 > example_fulco2019/input_data/wgEncodeUwDnaseK562Pk.500bp.bed
# remove blacklisted 
bedtools subtract -A -a  example_fulco2019/input_data/wgEncodeUwDnaseK562Pk.500bp.bed -b reference/hg19-blacklist.bed > example_fulco2019/input_data/wgEncodeUwDnaseK562Pk.filtered.bed
# add TSS 500bp
cat example_fulco2019/input_data/wgEncodeUwDnaseK562Pk.filtered.bed reference/RefSeqCurated.170308.bed.CollapsedGeneBounds.TSS500bp.bed |  awk -F"\t" '{print $1"\t"$2"\t"$3}' |  sort -k1,1 -k2,2n  | uniq > example_fulco2019/input_data/wgEncodeUwDnaseK562Pk.TSS.bed
# merge 
bedtools merge -i example_fulco2019/input_data/wgEncodeUwDnaseK562Pk.TSS.bed > example_fulco2019/input_data/wgEncodeUwDnaseK562Pk.merged.bed
ln -s  example_fulco2019/input_data/wgEncodeUwDnaseK562Pk.merged.bed  example_fulco2019/input_data/Fulco2019.candidate.enhancer.bed
# total 161802 candidate regions average length 575.5 


conda activate final-abc-env
#Quantifying Enhancer Activity
python src/run.neighborhoods.py \
--candidate_enhancer_regions example_fulco2019/input_data/Fulco2019.candidate.enhancer.bed \
--genes reference/RefSeqCurated.170308.bed.CollapsedGeneBounds.bed \
--H3K27ac example_fulco2019/input_data/ENCFF384ZZM.bam \
--DHS example_fulco2019/input_data/wgEncodeUwDnaseK562AlnRep1.bam,example_fulco2019/input_data/wgEncodeUwDnaseK562AlnRep2.bam \
--expression_table example_fulco2019/input_data/K562.ENCFF934YBO.TPM.txt \
--chrom_sizes reference/chr_sizes \
--ubiquitously_expressed_genes reference/UbiquitouslyExpressedGenesHG19.txt \
--cellType K562 \
--outdir example_fulco2019/ABC_output/Neighborhoods/ 


#Computing the ABC Score
python src/predict.py \
--enhancers example_fulco2019/ABC_output/Neighborhoods/EnhancerList.txt \
--genes example_fulco2019/ABC_output/Neighborhoods/GeneList.txt \
--HiCdir example_fulco2019/input_data/HiC/raw/ \
--chrom_sizes reference/chr_sizes \
--hic_resolution 5000 \
--scale_hic_using_powerlaw \
--threshold .02 \
--cellType K562 \
--outdir example_fulco2019/ABC_output/Predictions/ \
--make_all_putative \
--chromosome chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX

