#!/usr/bin/bash
set -uex

DATADIR=data/Gasperini
#DATADIR=$DATADIR.newTFs
CRISPRFILE=$DATADIR/Gasperini2019.at_scale_screen.cand_enhancer_x_exprsd_genes.200503.csv
ABCOUTDIR=/data8/han_lab/mhan/ABC-Enhancer-Gene-Prediction/example_fulco2019/ABC_output/
TFFILE=data/ucsc/encRegTfbsClusteredWithK562.hg19.bed
HECKERDIR=../Hecker_CRISPRi_screens
#TFFILE=/data8/han_lab/mhan/abic/data/encodeTF2/consolidatedEncodeTFBS.bed
awk -F"," '{print $3"\t"$4"\t"$5"\t"$2}' $CRISPRFILE  | sed 's/"//g' | awk -F":" 'NR>1 {print $1}' | sort -k1,1 -k2,2n -k3,3n | uniq > $DATADIR/Gasperini2019.enhancer.bed
awk -F"," '{print $6"\t"$7-500"\t"$8+500"\t"$14"\t0\t"$10}' $CRISPRFILE | sed 's/"//g' | awk 'NR>1' | sort -k1,1 -k2,2n -k3,3n | uniq > $DATADIR/Gasperini2019.TSS.bed
awk '{print $4}' $DATADIR/Gasperini2019.TSS.bed | sort | uniq > $DATADIR/Gasperini2019.TSS.symbol.sorted
join $HECKERDIR/Gasperini_interactions_wScore.genes.symbol.sorted $DATADIR/Gasperini2019.TSS.symbol.sorted > $DATADIR/Gasperini2019.Hecker.TSS.symbol.sorted
grep -w -f $DATADIR/Gasperini2019.Hecker.TSS.symbol.sorted $DATADIR/Gasperini2019.TSS.bed > $DATADIR/Gasperini2019.hecker.TSS.bed

bedtools intersect -wb -a $HECKERDIR/Gasperini_interactions_wScore.enhancers.bed -b $DATADIR/Gasperini2019.enhancer.bed | awk '{print $5"\t"$6"\t"$7"\t"$8}'  > $DATADIR/Gasperini2019.hecker.enhancer.bed

echo -e "G.chr\tG.start\tG.end\tG.id\tABC.chr\tABC.start\tABC.end\tABC.class\tABC.id\toverlap" > $DATADIR/Gasperini2019.enhancer.ABC.overlap.bed

#bedtools intersect -wo -e -f 0.3 -F 0.3 -a $DATADIR/Gasperini2019.enhancer.bed -b $ABCOUTDIR/Neighborhoods/EnhancerList.bed | sed 's/|/\t/' >> $DATADIR/Gasperini2019.enhancer.ABC.overlap.bed
bedtools intersect -wb -a $HECKERDIR/Hecker_RegularABC_candidate_enhancers.bed -b $ABCOUTDIR/Neighborhoods/EnhancerList.bed | awk '{print $5"\t"$6"\t"$7"\t"$8}' > $DATADIR/EnhancerList.bed
bedtools intersect -wo -e -f 0.3 -F 0.3 -a $DATADIR/Gasperini2019.hecker.enhancer.bed -b $DATADIR/EnhancerList.bed | sed 's/|/\t/' >> $DATADIR/Gasperini2019.enhancer.ABC.overlap.bed
 
awk '{print $4"\t"$0}' $DATADIR/Gasperini2019.hecker.TSS.bed | sort -k1,1 > /tmp/Gasperini.TSS
awk '{print $4"\t"$0}' $HECKERDIR/Hecker_RegularABC_candidate_genes.bed | sort -k1,1 > /tmp/ABC.TSS
echo -e "G.chr\tG.start\tG.end\tgene\tscore\tstrand\tABC.chr\tABC.start\tABC.end" > $DATADIR/Gasperini2019.TSS.ABC.overlap.bed
join -a 1 /tmp/Gasperini.TSS /tmp/ABC.TSS  | awk '{print $2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10}' >> $DATADIR/Gasperini2019.TSS.ABC.overlap.bed

awk '{print $1":"$2"-"$3"_"$7"\t"$0}' $ABCOUTDIR/Predictions/EnhancerPredictionsAllPutativeExpNonExpGenes.txt > $DATADIR/ABC.EnhancerPredictionsAllPutative.txt


# TF
bedtools intersect -wo -e -f 0.3 -F 0.3 -a $DATADIR/Gasperini2019.hecker.enhancer.bed -b $TFFILE | sed 's/|/\t/' >> $DATADIR/Gasperini2019.enhancer.TF.overlap.bed
bedtools intersect -wo -e -f 0.3 -F 0.3 -a $DATADIR/Gasperini2019.hecker.TSS.bed -b $TFFILE | sed 's/|/\t/' >> $DATADIR/Gasperini2019.TSS.TF.overlap.bed


ln -s $ABCOUTDIR/Neighborhoods/EnhancerList.txt ${DATADIR}/EnhancerList.txt 
ln -s $ABCOUTDIR/Neighborhoods/GeneList.txt ${DATADIR}/GeneList.txt 

ln -s $ABCOUTDIR/Neighborhoods.H3K4me3/EnhancerList.txt ${DATADIR}/EnhancerList.H3K4me3.txt 
ln -s $ABCOUTDIR/Neighborhoods.H3K4me3/GeneList.txt ${DATADIR}/GeneList.H3K4me3.txt 

ln -s $ABCOUTDIR/Neighborhoods.H3K27me3/EnhancerList.txt ${DATADIR}/EnhancerList.H3K27me3.txt 
ln -s $ABCOUTDIR/Neighborhoods.H3K27me3/GeneList.txt ${DATADIR}/GeneList.H3K27me3.txt 



# find overlap with ABC, TF
python src/preprocess/overlap.gasperini.py




# apply DR
python src/preprocess/applynmf.py --dir ${DATADIR} --infile Gasperini2019.at_scale.ABC.TF.erole.txt --NMFdir data/Gasperini.fixed/


# drop NA from pValueAdjusted
python src/preprocess/dropna.py --dir $DATADIR/ --infile Gasperini2019.at_scale.ABC.TF.erole.test.txt



# extract rows with genes that have at least 1 significant enhancer
python src/preprocess/atleast1sig.py --dir $DATADIR/ --infile Gasperini2019.at_scale.ABC.TF.erole.test.dropna.txt


