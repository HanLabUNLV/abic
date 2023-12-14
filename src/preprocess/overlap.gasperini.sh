#!/usr/bin/bash
set -uex

DATADIR=data/Gasperini
CRISPRFILE=/data8/han_lab/mhan/abic/$DATADIR/Gasperini2019.at_scale_screen.cand_enhancer_x_exprsd_genes.200503.csv
ABCOUTDIR=/data8/han_lab/mhan/ABC-Enhancer-Gene-Prediction.bak/Gasperini/ABC_output
TFFILE=/data8/han_lab/mhan/abic/data/ucsc/encRegTfbsClusteredWithK562.hg19.bed
awk -F"," '{print $3"\t"$4"\t"$5"\t"$2}' $CRISPRFILE  | sed 's/"//g' | awk -F":" 'NR>1 {print $1}' | sort -k1,1 -k2,2n -k3,3n | uniq > $DATADIR/Gasperini2019.enhancer.bed
awk -F"," '{print $6"\t"$7-500"\t"$8+500"\t"$14"\t0\t"$10}' $CRISPRFILE | sed 's/"//g' | awk 'NR>1' | sort -k1,1 -k2,2n -k3,3n | uniq > $DATADIR/Gasperini2019.TSS.bed

echo -e "G.chr\tG.start\tG.end\tG.id\tABC.chr\tABC.start\tABC.end\tABC.class\tABC.id\toverlap" > $DATADIR/Gasperini2019.enhancer.ABC.overlap.bed
bedtools intersect -wo -e -f 0.3 -F 0.3 -a $DATADIR/Gasperini2019.enhancer.bed -b $ABCOUTDIR/Neighborhoods/EnhancerList.bed | sed 's/|/\t/' >> $DATADIR/Gasperini2019.enhancer.ABC.overlap.bed
 
awk '{print $4"\t"$0}' $DATADIR/Gasperini2019.TSS.bed | sort -k1,1 > /tmp/Gasperini.TSS
awk '{print $4"\t"$0}' $ABCOUTDIR/Neighborhoods/GeneList.TSS1kb.bed | sort -k1,1 > /tmp/ABC.TSS
echo -e "G.chr\tG.start\tG.end\tgene\tscore\tstrand\tABC.chr\tABC.start\tABC.end" > $DATADIR/Gasperini2019.TSS.ABC.overlap.bed
join -a 1 /tmp/Gasperini.TSS /tmp/ABC.TSS  | awk '{print $2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10}' >> $DATADIR/Gasperini2019.TSS.ABC.overlap.bed

awk '{print $1":"$2"-"$3"_"$7"\t"$0}' $ABCOUTDIR/Predictions/EnhancerPredictionsAllPutative.txt > $DATADIR/ABC.EnhancerPredictionsAllPutative.txt


# TF
bedtools intersect -wo -e -f 0.3 -F 0.3 -a $DATADIR/Gasperini2019.enhancer.bed -b $TFFILE | sed 's/|/\t/' >> $DATADIR/Gasperini2019.enhancer.TF.overlap.bed
bedtools intersect -wo -e -f 0.3 -F 0.3 -a $DATADIR/Gasperini2019.TSS.bed -b $TFFILE | sed 's/|/\t/' >> $DATADIR/Gasperini2019.TSS.TF.overlap.bed


#ln -s $ABCOUTDIR/Neighborhoods/EnhancerList.txt ${DATADIR}/EnhancerList.txt 
#ln -s $ABCOUTDIR/Neighborhoods/GeneList.txt ${DATADIR}/GeneList.txt 
#
#ln -s $ABCOUTDIR/Neighborhoods.H3K4me3/EnhancerList.txt ${DATADIR}/EnhancerList.H3K4me3.txt 
#ln -s $ABCOUTDIR/Neighborhoods.H3K4me3/GeneList.txt ${DATADIR}/GeneList.H3K4me3.txt 
#
#ln -s $ABCOUTDIR/Neighborhoods.H3K27me3/EnhancerList.txt ${DATADIR}/EnhancerList.H3K27me3.txt 
#ln -s $ABCOUTDIR/Neighborhoods.H3K27me3/GeneList.txt ${DATADIR}/GeneList.H3K27me3.txt 



# find overlap with ABC, TF
python src/preprocess/overlap.gasperini.py

# calculate indirect ABC scores
#python src/network/calculate_abic.py  --netdir data/epgraph.Gasperini.K562/ --dir data/Gasperini/ --infile Gasperini2019.at_scale.ABC.TF.erole.txt  

# extract rows with genes that have at least 1 significant enhancer
#python src/preprocess/atleast1sig.py --dir data/Gasperini/ --infile Gasperini2019.at_scale.ABC.TF.erole.ABCpath.txt 

# group by chromosomal position for groupCV
python src/preprocess/groupbypos.py --dir data/Gasperini/ --infile Gasperini2019.at_scale.ABC.TF.erole.txt 


# split train-test and fit DR (NMF) to train
python src/preprocess/split_test_dr_fitnmf.py --dir data/Gasperini/ --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.txt

# apply DR(NMF) to test
python src/preprocess/applynmf.py --dir data/Gasperini --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.test.txt --NMFdir data/Gasperini/


# split data by complexity 
python src/preprocess/complexity.py --dir data/Gasperini/ --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.train.txt 
python src/preprocess/complexity.py --dir data/Gasperini/ --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.test.txt --target Gasperini2019.at_scale.ABC.TF.erole.grouped.test.target.txt 



# generate train and test for gene prediction
grep -v 'chr5' $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.txt | grep -v 'chr10' | grep -v 'chr15' | grep -v 'chr20' > $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.train.txt

head -n 1 $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.train.txt > $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.test.txt
grep 'chr5' $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.txt  >> $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.test.txt
grep 'chr10' $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.txt  >> $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.test.txt 
grep 'chr15' $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.txt  >> $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.test.txt
grep 'chr20' $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.txt  >> $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.test.txt

awk -F"\t"  '{print $7}' $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.test.txt > $DATADIR/Gasperini2019.bygene.ABC.TF.grouped.test.target.txt



