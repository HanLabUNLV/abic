#!/usr/bin/bash
set -uex

CRISPRFILE=/data8/han_lab/mhan/abic/data/Fulco/Fulco2019.STable6a.tab
ABCOUTDIR=/data8/han_lab/mhan/ABC-Enhancer-Gene-Prediction/example_fulco2019/ABC_output
TFFILE=/data8/han_lab/mhan/abic/data/ucsc/encRegTfbsClusteredWithK562.hg19.bed


awk -F"\t" '{print $1"\t"$2"\t"$3"\t"$4}' $CRISPRFILE | sort -k1,1 -k2,2n -k3,3n | uniq > data/Fulco/Fulco2019.enhancer.bed
awk -F"\t" '{print $1"\t"$18-500"\t"$18+500"\t"$5"\t0\t."}' $CRISPRFILE | sed 's/"//g' | sort -k1,1 -k2,2n -k3,3n | uniq > data/Fulco/Fulco2019.TSS.bed

echo -e "G.chr\tG.start\tG.end\tG.id\tABC.chr\tABC.start\tABC.end\tABC.id\toverlap" > data/Fulco/Fulco2019.enhancer.ABC.overlap.bed
bedtools intersect -wo -e -f 0.3 -F 0.3 -a data/Fulco/Fulco2019.enhancer.bed -b $ABCOUTDIR/Neighborhoods/EnhancerList.bed  >> data/Fulco/Fulco2019.enhancer.ABC.overlap.bed
 

awk '{print $4"\t"$0}' data/Fulco/Fulco2019.TSS.bed | sort -k1,1 > /tmp/Fulco.TSS
awk '{print $4"\t"$0}' $ABCOUTDIR/Neighborhoods/GeneList.TSS1kb.bed | sort -k1,1 > /tmp/ABC.TSS
echo -e "G.chr\tG.start\tG.end\tgene\tscore\tstrand\tABC.chr\tABC.start\tABC.end" > data/Fulco/Fulco2019.TSS.ABC.overlap.bed
join -a 1 /tmp/Fulco.TSS /tmp/ABC.TSS  | awk '{print $2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10}' >> data/Fulco/Fulco2019.TSS.ABC.overlap.bed

awk '{print $1":"$2"-"$3"_"$7"\t"$0}' $ABCOUTDIR/Predictions/EnhancerPredictionsAllPutative.txt > data/Fulco/ABC.EnhancerPredictionsAllPutative.txt



# TF
bedtools intersect -wo -e -f 0.3 -F 0.3 -a data/Fulco/Fulco2019.enhancer.bed -b $TFFILE  | sed 's/|/\t/' >> data/Fulco/Fulco2019.enhancer.TF.overlap.bed
bedtools intersect -wo -e -f 0.3 -F 0.3 -a data/Fulco/Fulco2019.TSS.bed -b $TFFILE  | sed 's/|/\t/' >> data/Fulco/Fulco2019.TSS.TF.overlap.bed


ln -s $ABCOUTDIR/Neighborhoods/EnhancerList.txt data/Fulco/EnhancerList.txt 
ln -s $ABCOUTDIR/Neighborhoods/GeneList.txt data/Fulco/GeneList.txt 

ln -s $ABCOUTDIR/Neighborhoods.H3K4me3/EnhancerList.txt data/Fulco/EnhancerList.H3K4me3.txt 
ln -s $ABCOUTDIR/Neighborhoods.H3K4me3/GeneList.txt data/Fulco/GeneList.H3K4me3.txt 

ln -s $ABCOUTDIR/Neighborhoods.H3K27me3/EnhancerList.txt data/Fulco/EnhancerList.H3K27me3.txt 
ln -s $ABCOUTDIR/Neighborhoods.H3K27me3/GeneList.txt data/Fulco/GeneList.H3K27me3.txt 

# run overlap.fulco.py
python src/preprocess/overlap.fulco.py 

# generate target.txt
awk -F"\t" '{print $797}' data/Fulco/Fulco2019.CRISPR.ABC.TF.cobinding.txt > data/Fulco/Fulco2019.CRISPR.ABC.TF.cobinding.target.txt
#
