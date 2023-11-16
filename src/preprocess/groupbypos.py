import pandas as pd
import numpy as np

datadir="/data8/han_lab/mhan/abic/data/Gasperini/"
df = pd.read_csv(datadir+'Gasperini2019.at_scale.ABC.TF.cobinding.erole.txt', sep="\t", index_col=0)
#del df[df.columns[0]]
#del df[df.columns[0]]

#chrs = [y for x, y in df.groupby('chrEnhancer')]
df_by_chr = {chr_v: df[df['chrEnhancer'] == chr_v].copy() for chr_v in df.chrEnhancer.unique()}

for chromosome in df_by_chr.keys():
  chr_df = df_by_chr[chromosome]
  chr_df['midbin'] = round(chr_df[['startEnhancer', 'endEnhancer']].mean(axis=1)/1000000)
  chr_df.sort_values(['startEnhancer', 'startTSS'], ascending=[True, True], inplace=True, ignore_index=True)
  chr_df['group'] = 'NA'
  binshift = list(chr_df['midbin'].iloc[1:])
  binshift.append(binshift[len(binshift)-1])  
  # find gaps with > 5000000
  gaps = chr_df.copy()
  gaps['nextbin'] = binshift
  gaps['bingap'] = gaps['nextbin']-gaps['midbin']
  gaps = gaps.loc[gaps['bingap']>5,]
  # find boundary bind number for gaps
  gapboundary = list(gaps['midbin']+1)
  gapboundary.insert(0, min(chr_df['midbin'])-1)
  gapboundary.append(max(chr_df['midbin'])+1)
  # set group ID based on gap boundaries 
  for i in range(1, len(gapboundary)):
    chr_df.loc[(chr_df['midbin']>gapboundary[i-1]) & (chr_df['midbin']<gapboundary[i]), 'group'] = chromosome+'.'+str(i)


  newdf = pd.concat(df_by_chr.values(), axis=0)
  newdf.to_csv(datadir+"Gasperini2019.at_scale.ABC.TF.cobinding.erole.grouped.txt", sep="\t")

