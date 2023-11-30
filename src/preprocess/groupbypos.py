import pandas as pd
import numpy as np


def group_by_pos(arg_df):
  arg_df['midbin'] = round(arg_df[['startEnhancer', 'endEnhancer']].mean(axis=1)/1000000)
  arg_df.sort_values(['startEnhancer', 'startTSS'], ascending=[True, True], inplace=True, ignore_index=True)
  arg_df['group'] = 'NA'
  binshift = list(arg_df['midbin'].iloc[1:])
  binshift.append(binshift[len(binshift)-1])  
  # find gaps with > 5000000
  gaps = arg_df.copy()
  gaps['nextbin'] = binshift
  gaps['bingap'] = gaps['nextbin']-gaps['midbin']
  gaps = gaps.loc[gaps['bingap']>5,]
  # find boundary bind number for gaps
  gapboundary = list(gaps['midbin']+1)
  gapboundary.insert(0, min(arg_df['midbin'])-1)
  gapboundary.append(max(arg_df['midbin'])+1)
  # set group ID based on gap boundaries 
  for i in range(1, len(gapboundary)):
    arg_df.loc[(arg_df['midbin']>gapboundary[i-1]) & (arg_df['midbin']<gapboundary[i]), 'group'] = chromosome+'.'+str(i)
  return arg_df



datadir="/data8/han_lab/mhan/abic/data/Gasperini/"
df = pd.read_csv(datadir+'Gasperini2019.at_scale.ABC.TF.NMF.erole.txt', sep="\t", index_col=0)
df_by_chr = {chr_v: df[df['chrEnhancer'] == chr_v].copy() for chr_v in df.chrEnhancer.unique()}

newdf = pd.DataFrame()
for chromosome in df_by_chr.keys():
  chr_df = df_by_chr[chromosome].copy()
  new_chr_df = group_by_pos(chr_df)
  newdf = pd.concat([newdf, new_chr_df], axis=0)
newdf.to_csv(datadir+"Gasperini2019.at_scale.ABC.TF.NMF.erole.grouped.txt", sep="\t")

df = pd.read_csv(datadir+'Gasperini2019.at_scale.ABC.TF.NMF.erole.atleast1sig.txt', sep="\t", index_col=0)
df_by_chr = {chr_v: df[df['chrEnhancer'] == chr_v].copy() for chr_v in df.chrEnhancer.unique()}


newdf = pd.DataFrame()
for chromosome in df_by_chr.keys():
  chr_df = df_by_chr[chromosome].copy()
  new_chr_df = group_by_pos(chr_df)
  newdf = pd.concat([newdf, new_chr_df], axis=0)
newdf.to_csv(datadir+"Gasperini2019.at_scale.ABC.TF.NMF.erole.grouped.atleast1sig.txt", sep="\t")

