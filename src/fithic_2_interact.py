import pandas as pd
import os 


cols = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'value', 'exp', 'color','sourceChrom', 'sourceStart', 'sourceEnd','sourceName','sourceStrand','targetChrom', 'targetStart', 'targetEnd','targetName','targetStrand']
bigdata = pd.DataFrame(columns=cols)


for chrm in os.listdir('data/fithic_loops/'):
    print(chrm)
    data = pd.read_csv('data/fithic_loops/'+chrm+'/fithic_filtered.bedpe', sep='\t')

    data.rename(columns={'chr1':'sourceChrom','start1':'sourceStart','end1':'sourceEnd','chr2':'targetChrom','start2':'targetStart','end2':'targetEnd','contactCount':'value'}, inplace=True)
    data.drop(['p-value', 'q-value', 'bias1', 'bias2'], axis=1, inplace=True)

    data['chrom']=data['sourceChrom']
    data['chromStart'] = data[['sourceStart','sourceEnd','targetStart','targetEnd']].min(axis=1)
    data['chromEnd'] = data[['sourceStart','sourceEnd','targetStart','targetEnd']].max(axis=1)
    data['name'] = '.'
    data['score'] = (data['value'] / data['value'].max()) * 1000
    data['exp'] = '.'
    data['color'] = 0
    data['sourceName']='.'
    data['targetName']='.'
    data['sourceStrand']='.'
    data['targetStrand']='.'

    data = data[cols]
    
    print(len(data))

    data.to_csv('data/interact_files/k562_'+chrm+'.bigInteract', sep='\t')

