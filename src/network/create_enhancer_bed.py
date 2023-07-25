#start with the command
#cut -f1,2,3 /data8/han_lab/dbarth/ncbi/public/jonathan/han_rep/Predictions/EnhancerPredictionsAllPutative.txt | uniq > PutativeEnhancer.bed

#this is all the enhancers, but we still need promoters
gene_tss = {}
with open('data/gene_tss.subset1.tsv','r') as f:
    for line in f:
        chromosome, gene, tss = line.strip().split('\t')
        gene_tss[gene] = [chromosome, int(tss)]

lines = ['\t'.join([gene_tss[gene][0], str(gene_tss[gene][1]-500), str(gene_tss[gene][1]+500)]) for gene in gene_tss]

with open('data/gas_enhancers.uniq.bed','a') as f:
    for line in lines:
        f.write(line+'\n')


