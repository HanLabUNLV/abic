import pandas as pd
import pyranges as pr

#import data, isolate sig e2
data = pd.read_csv('data/Gasperini2019.at_scale.ABC.TF.roles.txt',sep='\t')
data = data.rename(columns={'chrEnhancer':'Chromosome', 'startEnhancer':'Start','endEnhancer':'End'})
#e2_sig = data.loc[(data['e2']==1)&(data['Significant']==True),]
e2_sig = data.loc[(data['e2']==1),]


#add columns for new tfs from e1
tfs = [x.split('_')[0] for x in e2_sig.columns.tolist() if '_e' in x]
for tf in tfs:
    e2_sig[tf+'_e1'] = 0
    e2_sig = e2_sig.rename(columns={tf+'_e':tf+'_e2'})

#init output df with all the right columns
outdf = pd.DataFrame(columns=e2_sig.columns.tolist() + ['start_e1','end_e1','Significant_e1', 'activity_e1', 'hic_contact_e1'])
outdf = outdf.rename(columns={'Chromosome':'chr','Significant':'Significant_e2','Start':'start_e2','End':'end_e2', 'activity_base_x':'activity_e2','hic_contact':'hic_contact_e2','TargetGene':'gene'})
outdf = outdf.drop(['Notes', 'ReadoutMethod', 'chrTSS', 'chr:start-end_TargetGene', 'hic_contact_pl_scaled', 'EffectSize', 'Unnamed: 0.1', 'pValue', 'powerlaw.Score.Numerator', 'start_y', 'enhancerID', 'GeneSymbol', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'EnhancerID', 'normalized_dhs', 'ABC.Score.Numerator', 'hic_contact_pl_scaled_adj', 'end_x', 'PerturbationMethod', 'e1', 'ABC.Score', 'powerlaw_contact', 'endTSS', 'hic_pseudocount', 'GenomeBuild', 'TargetGeneIsExpressed', 'Unnamed: 0', 'strandGene', 'powerlaw_contact_reference', 'chr_x', 'EnsemblD', 'ABC.id', 'startTSS', 'powerlaw.Score', 'name', 'distance', 'activity_base_y', 'name_y', 'e3', 'CellType_y', 'start_x', 'Reference', 'G.id', 'TargetGeneTSS', 'class_x', 'TargetGenePromoterActivityQuantile', 'class_y', 'TargetGeneExpression', 'name_x', 'CellType_x', 'isSelfPromoter', 'normalized_h3K27ac', 'e2', 'end_y', 'pValueAdjusted', 'gRNAs', 'chr_y'],axis=1)

#look up e2 (e2 + gene) coords in e1-e2 pairs
#then append all e1 tf cols as well as significant as _e1 and change _e to _e2
pairs = pd.read_csv('data/e1_e2_validated_pairs_trimmed.tsv',sep='\t')
pairs['chr_e1'],pairs['start_e1'],pairs['end_e1'] = pairs['e1'].str.split('_',2).str
pairs['Chromosome'],pairs['Start'],pairs['End'] = pairs['e2'].str.split('_',2).str
pairs['Start'] = pairs['Start'].astype(int)
pairs['End'] = pairs['End'].astype(int)
prpairs = pr.PyRanges(pairs)
prdata = pr.PyRanges(data)
pre2 = pr.PyRanges(e2_sig.rename(columns={'chr':'Chromosome','start':'Start','end':'End'}))
fump=0
for idx, row in e2_sig.iterrows():
    #create temp pyranges to find which e2-e1 connections are relevant 
    tmppr = pr.from_dict({'Chromosome':[row['Chromosome']],'Start':[row['Start']], 'End':[row['End']]})
    ov = prpairs.overlap(tmppr).as_df() #overlaps all coords
    #ov contains e2 in primary coords, _e1 in secondary
    
    if len(ov)>0:
        e1s = ov.loc[ov['gene']==row['GeneSymbol']] #narrows to just this gene
    
        #the e1s are in their 5kb coordinates, so we need to reoverlap them with the e1s from the data
        for idy, roy in e1s.iterrows():
            tmppr = pr.from_dict({'Chromosome':[roy['chr_e1']],'Start':[int(roy['start_e1'])], 'End':[int(roy['end_e1'])]})
            pov = prdata.overlap(tmppr).as_df()
            if len(pov)>0:
                pov = pov.loc[pov['GeneSymbol']==roy['gene']]

                #select some variables and create a new row to store data
                extract_cols = [i+'_e' for i in tfs]
                extract_cols.extend(['Significant', 'Chromosome','Start','End'])
                newrow = pov[extract_cols]
                
                #append to outdf
                #print(row)
                #print(newrow)
                for tf in tfs:
                    newrow = newrow.rename(columns={tf+'_e':tf+'_e1'})
                    newrow[tf+'_e2'] = row[tf+'_e2']
                    newrow[tf+'_TSS'] = row[tf+'_TSS']

                newrow = newrow.rename(columns={'Chromosome':'chr','Start':'start_e1','End':'end_e1', 'Significant':'Significant_e1', 'activity_base_x':'activity_e1', 'hic_contact':'hic_contact_e1'})
                
                newrow['Significant_e2'] = row['Significant']
                newrow['start_e2'] = row['Start']
                newrow['end_e2'] = row['End']

                newrow['start_e1'] = pov['Start']                        
                newrow['end_e1'] = pov['End']                        

                newrow['activity_e1'] = pov['activity_base_x']
                newrow['activity_e2'] = row['activity_base_x']
                newrow['hic_contact_e1'] = pov['hic_contact'] 
                newrow['hic_contact_e2'] = row['hic_contact'] 
                newrow['gene'] = row['GeneSymbol']
                outdf = pd.concat([outdf,newrow], axis=0, ignore_index=True)
    print(len(outdf))
print(outdf)
outdf.to_csv('data/e2_e1_pairs_binding.tsv',sep='\t')
exit()

