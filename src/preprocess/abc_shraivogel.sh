##juicer_tools
#wget http://hicfiles.tc4ga.com.s3.amazonaws.com/public/juicer/juicer_tools.1.7.5_linux_x64_jcuda.0.8.jar
#ln -s juicer_tools.1.7.5_linux_x64_jcuda.0.8.jar juicer_tools.jar
#
#gather data
cd example_shraivogel/input_data
# DNAse narrowpeak
wget https://www.encodeproject.org/files/ENCFF623AFX/@@download/ENCFF623AFX.bed.gz
wget https://www.encodeproject.org/files/ENCFF494DBY/@@download/ENCFF494DBY.bed.gz
wget https://raw.githubusercontent.com/Boyle-Lab/Blacklist/master/lists/Blacklist_v1/hg19-blacklist.bed.gz
#
# DNAse
wget https://www.encodeproject.org/files/ENCFF441RET/@@download/ENCFF441RET.bam
wget https://www.encodeproject.org/files/ENCFF271LGJ/@@download/ENCFF271LGJ.bam
# H3K27ac
wget https://www.encodeproject.org/files/ENCFF384ZZM/@@download/ENCFF384ZZM.bam #rep1
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



##
#conda activate final-abc-env
samtools index ENCFF441RET.bam
samtools index ENCFF271LGJ.bam
samtools index ENCFF384ZZM.bam
#cd ../..
cp example_chr22/input_data/Expression/K562.ENCFF934YBO.TPM.txt example_shraivogel/input_data/

##Download hic matrix file  juicebox
#python src/juicebox_dump.py \
#--hic_file https://hicfiles.s3.amazonaws.com/hiseq/k562/in-situ/combined_30.hic \
#--juicebox "java -jar juicer_tools.jar" \
#--outdir example_shraivogel/input_data/HiC/raw/ 
#
##Fit HiC data to powerlaw model and extract parameters
#python src/compute_powerlaw_fit__hic.py \
#--hicDir example_shraivogel/input_data/HiC/raw/ \
#--outDir example_shraivogel/input_data/HiC/raw/powerlaw/ \
#--maxWindow 1000000 \
#--minWindow 5000 \
#--resolution 5000 
#
cp -r example_wg/input_data/HiC example_shraivogel/input_data/


##Define candidate enhancer regions
# 
conda activate final-abc-env 
# get candidate regions from meta_data/allEnhancers.hg38.txt
cp example_shraivogel/input_data/allEnhancers.hg19.bed example_shraivogel/input_data/enhancers_from_shraivogel.bed

# expand to make 500bp
TARGET_LENGTH=500
awk -vF=${TARGET_LENGTH} 'BEGIN{ OFS="\t"; }{ len=$3-$2; diff=F-len; flank=int(diff/2); upflank=downflank=flank; if (diff%2==1) { downflank++; }; if(diff<0) {upflank=0; downflank=0;}; print $1, $2-upflank, $3+downflank; }' example_shraivogel/input_data/enhancers_from_shraivogel.bed > example_shraivogel/input_data/enhancers.500bp.bed

# remove blacklisted 
bedtools subtract -A -a  example_shraivogel/input_data/enhancers.500bp.bed -b reference/wgEncodeHg19ConsensusSignalArtifactRegions.bed > example_shraivogel/input_data/enhancers.filtered.bed
# add TSS 500bp
cat example_shraivogel/input_data/enhancers.filtered.bed reference/RefSeqCurated.170308.bed.CollapsedGeneBounds.TSS500bp.bed |  awk -F"\t" '{print $1"\t"$2"\t"$3}' |  sort -k1,1 -k2,2n  | uniq > example_shraivogel/input_data/enhancers.TSS.bed
# merge 
bedtools merge -i example_shraivogel/input_data/enhancers.TSS.bed > example_shraivogel/input_data/enhancers.merged.bed
ln -s  example_shraivogel/input_data/enhancers.merged.bed  example_shraivogel/input_data/Shraivogel.candidate.enhancer.bed
# total 68660 candidate regions average length 600.1


conda activate final-abc-env
#Quantifying Enhancer Activity
python src/run.neighborhoods.py \
--candidate_enhancer_regions example_shraivogel/input_data/Shraivogel.candidate.enhancer.bed \
--genes reference/RefSeqCurated.170308.bed.CollapsedGeneBounds.bed \
--H3K27ac example_shraivogel/input_data/ENCFF384ZZM.bam \
--DHS example_shraivogel/input_data/ENCFF441RET.bam,example_shraivogel/input_data/ENCFF271LGJ.bam \
--expression_table example_shraivogel/input_data/K562.ENCFF934YBO.TPM.txt \
--chrom_sizes reference/chr_sizes \
--ubiquitously_expressed_genes reference/UbiquitouslyExpressedGenesHG19.txt \
--cellType K562 \
--outdir example_shraivogel/ABC_output/Neighborhoods/ 


#Computing the ABC Score
python src/predict.py \
--enhancers example_shraivogel/ABC_output/Neighborhoods/EnhancerList.txt \
--genes example_shraivogel/ABC_output/Neighborhoods/GeneList.txt \
--HiCdir example_shraivogel/input_data/HiC/raw/ \
--chrom_sizes reference/chr_sizes \
--hic_resolution 5000 \
--scale_hic_using_powerlaw \
--threshold .02 \
--cellType K562 \
--outdir example_shraivogel/ABC_output/Predictions/ \
--make_all_putative \
--chromosome chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX

