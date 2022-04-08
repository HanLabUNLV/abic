from os import listdir as ls
import pickle as pkl

enhancers = []
for gfile in ls('data/gene_networks_wd/'):
    network = pkl.load(open('data/gene_networks_wd/'+ gfile, 'rb'))
    for n in network.vs:
        if n['enhancers'] is not None:
            for enh in n['enhancers']['local_enhancers']:
                eid = '\t'.join([str(x) for x in enh])
                if eid not in enhancers:
                    enhancers.append(eid)

#this is all the enhancers, but we still need promoters
gene_tss = {}
with open('data/gene_tss.subset1.tsv','r') as f:
    for line in f:
        chromosome, gene, tss = line.strip().split('\t')
        gene_tss[gene] = [chromosome, int(tss)]

promoters = ['\t'.join([gene_tss[gene][0], str(gene_tss[gene][1]-500), str(gene_tss[gene][1]+500)]) for gene in gene_tss]

enhancers.extend(promoters)
with open('data/subset1_enhancers.bed','a') as f:
    for line in enhancers:
        f.write(line+'\n')


