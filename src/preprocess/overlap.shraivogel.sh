#!/usr/bin/bash
set -uex

DATADIR=data/Shraivogel
CRISPRFILE=/data8/han_lab/mhan/abic/${DATADIR}/Shraivogel.input.txt
ABCOUTDIR=/data8/han_lab/mhan/ABC-Enhancer-Gene-Prediction/example_shraivogel/ABC_output
TSSPOS=$ABCOUTDIR/../../reference/RefSeqCurated.170308.bed.CollapsedGeneBounds.TSS500bp.bed
TFFILE=/data8/han_lab/mhan/abic/data/ucsc/encRegTfbsClusteredWithK562.hg19.bed


awk -F"\t" '{print $1"\t"$2"\t"$3"\t"$4}' $CRISPRFILE | sort -k1,1 -k2,2n -k3,3n | uniq > ${DATADIR}/enhancer.bed
awk '{print $5}' $CRISPRFILE | sort | uniq > genenames.txt
awk '{print $4"\t"$0}' $TSSPOS | LC_ALL=C sort > TSSpos.txt
LC_ALL=C join genenames.txt TSSpos.txt | awk '{print $2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7}' | bedtools sort -i > ${DATADIR}/TSS.bed

echo -e "G.chr\tG.start\tG.end\tG.id\tABC.chr\tABC.start\tABC.end\tABC.id\toverlap" > ${DATADIR}/enhancer.ABC.overlap.bed
bedtools intersect -wo -e -f 0.3 -F 0.3 -a ${DATADIR}/enhancer.bed -b $ABCOUTDIR/Neighborhoods/EnhancerList.bed  >> ${DATADIR}/enhancer.ABC.overlap.bed
 

awk '{print $4"\t"$0}' ${DATADIR}/TSS.bed | sort -k1,1 > /tmp/TSS
awk '{print $4"\t"$0}' $ABCOUTDIR/Neighborhoods/GeneList.TSS1kb.bed | sort -k1,1 > /tmp/ABC.TSS
echo -e "G.chr\tG.start\tG.end\tgene\tscore\tstrand\tABC.chr\tABC.start\tABC.end" > ${DATADIR}/TSS.ABC.overlap.bed
join -a 1 /tmp/TSS /tmp/ABC.TSS  | awk '{print $2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10}' >> ${DATADIR}/TSS.ABC.overlap.bed

awk '{print $1":"$2"-"$3"_"$7"\t"$0}' $ABCOUTDIR/Predictions/EnhancerPredictionsAllPutative.txt > ${DATADIR}/ABC.EnhancerPredictionsAllPutative.txt



# TF
bedtools intersect -wo -e -f 0.3 -F 0.3 -a ${DATADIR}/enhancer.bed -b $TFFILE  | sed 's/|/\t/' >> ${DATADIR}/enhancer.TF.overlap.bed
bedtools intersect -wo -e -f 0.3 -F 0.3 -a ${DATADIR}/TSS.bed -b $TFFILE  | sed 's/|/\t/' >> ${DATADIR}/TSS.TF.overlap.bed


ln -s $ABCOUTDIR/Neighborhoods/EnhancerList.txt ${DATADIR}/EnhancerList.txt 
ln -s $ABCOUTDIR/Neighborhoods/GeneList.txt ${DATADIR}/GeneList.txt 

ln -s $ABCOUTDIR/Neighborhoods.H3K4me3/EnhancerList.txt ${DATADIR}/EnhancerList.H3K4me3.txt 
ln -s $ABCOUTDIR/Neighborhoods.H3K4me3/GeneList.txt ${DATADIR}/GeneList.H3K4me3.txt 

ln -s $ABCOUTDIR/Neighborhoods.H3K27me3/EnhancerList.txt ${DATADIR}/EnhancerList.H3K27me3.txt 
ln -s $ABCOUTDIR/Neighborhoods.H3K27me3/GeneList.txt ${DATADIR}/GeneList.H3K27me3.txt 

# run overlap.shraivogel.py
python src/preprocess/overlap.shraivogel.py 



# apply DR
python src/preprocess/applynmf.py --dir ${DATADIR} --infile CRISPR.ABC.TF.txt --NMFdir data/Gasperini/


# extract rows with genes that have at least 1 significant enhancer
python src/preprocess/atleast1sig.py --dir ${DATADIR}/ --infile CRISPR.ABC.TF.test.txt



