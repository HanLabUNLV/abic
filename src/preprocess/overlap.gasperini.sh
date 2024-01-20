#!/usr/bin/bash
set -uex

DATADIR=data/Gasperini
#DATADIR=$DATADIR.newTFs
CRISPRFILE=$DATADIR/Gasperini2019.at_scale_screen.cand_enhancer_x_exprsd_genes.200503.csv
ABCOUTDIR=/scratch/han_lab/mhan/ABC-Enhancer-Gene-Prediction/Gasperini/
TFFILE=data/ucsc/encRegTfbsClusteredWithK562.hg19.bed
#TFFILE=/data8/han_lab/mhan/abic/data/encodeTF2/consolidatedEncodeTFBS.bed
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


ln -s $ABCOUTDIR/Neighborhoods/EnhancerList.txt ${DATADIR}/EnhancerList.txt 
ln -s $ABCOUTDIR/Neighborhoods/GeneList.txt ${DATADIR}/GeneList.txt 

ln -s $ABCOUTDIR/Neighborhoods.H3K4me3/EnhancerList.txt ${DATADIR}/EnhancerList.H3K4me3.txt 
ln -s $ABCOUTDIR/Neighborhoods.H3K4me3/GeneList.txt ${DATADIR}/GeneList.H3K4me3.txt 

ln -s $ABCOUTDIR/Neighborhoods.H3K27me3/EnhancerList.txt ${DATADIR}/EnhancerList.H3K27me3.txt 
ln -s $ABCOUTDIR/Neighborhoods.H3K27me3/GeneList.txt ${DATADIR}/GeneList.H3K27me3.txt 



# find overlap with ABC, TF
python src/preprocess/overlap.gasperini.py

# calculate indirect ABC scores
#python src/network/calculate_abic.py  --netdir data/epgraph.Gasperini.K562/ --dir $DATADIR/ --infile Gasperini2019.at_scale.ABC.TF.erole.txt  

# group by chromosomal position for groupCV
python src/preprocess/groupbypos.py --dir $DATADIR/ --infile Gasperini2019.at_scale.ABC.TF.erole.txt 

# split train-test and fit DR (NMF) to train
python src/preprocess/split_test_dr_fitnmf.py --dir $DATADIR/ --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.txt

# apply DR(NMF) to test
python src/preprocess/applynmf.py --dir $DATADIR --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.beforeNMF.txt --NMFdir $DATADIR/


# extract rows with genes that have at least 1 significant enhancer
python src/preprocess/atleast1sig.py --dir $DATADIR/ --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.train.txt
python src/preprocess/atleast1sig.py --dir $DATADIR/ --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.test.txt

# extract rows with genes that have greater than 2.5 TargetGeneExpression
python src/preprocess/hiexp.py --dir $DATADIR/ --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.train.txt
python src/preprocess/hiexp.py --dir $DATADIR/ --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.test.txt


# split data by complexity 
python src/preprocess/complexity.py --dir $DATADIR/ --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.train.txt 
python src/preprocess/complexity.py --dir $DATADIR/ --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.test.txt --target Gasperini2019.at_scale.ABC.TF.erole.grouped.test.target.txt 



# generate train and test for gene prediction
# group by chromosomal position
python src/preprocess/groupbypos.py --dir $DATADIR/ --infile Gasperini2019.bygene.ABC.TF.txt

# split train-test and fit DR (NMF) to train bygene data
python src/preprocess/split_test_dr_fitnmf.py --dir $DATADIR/ --infile Gasperini2019.bygene.ABC.TF.grouped.txt 

# apply DR(NMF) to test
python src/preprocess/applynmf.py  --dir $DATADIR/ --infile Gasperini2019.bygene.ABC.TF.grouped.beforeNMF.txt --NMFdir $DATADIR/ 

