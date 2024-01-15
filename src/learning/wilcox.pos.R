require(dplyr)
require(tidyr)
library(ggplot2)
library(ggpubr)

theme_set(theme_minimal(base_size = 14))
contact_threshold = 0.002 
ABC_threshold = 0.0021

ABC <- read.table("pos.gasperini.txt", header=TRUE, sep="\t", quote="")
weakpos <- ABC[ABC$hic_contact < contact_threshold,]
write.table(weakpos, "weakpos.gasperini.txt", sep="\t", quote=FALSE)
weakFNpos <- ABC[ABC$hic_contact <= contact_threshold & ABC$ABC.Score < ABC_threshold,]
write.table(weakFNpos, "weakFNpos.gasperini.txt", sep="\t", quote=FALSE)
weakFNpos <- weakFNpos[,c("ABC.id", "pValueAdjusted", "EffectSize", "ABC.Score", "hic_contact", "normalized_h3K27ac", "TSS.count.near.enhancer", "Enhancer.count.near.TSS", "remaining.TSS.contact.from.enhancer")]
weakFNpos <- weakFNpos[order(weakFNpos$ABC.id),]
write.table(weakFNpos, "weakFNpos.gasperini.txt", sep="\t", quote=FALSE)

ABC$strongcontact = ifelse(ABC$hic_contact >= contact_threshold, "strong", "weak")
ABC$siglabel = ifelse(ABC$Significant=="True", "pos", "neg")
ABC$case = as.factor(paste(ABC$strongcontact, ABC$siglabel))

ABC %>% 
  summarise_all(function(x) is.numeric(x)) %>%
  gather() %>%
  filter(value) %>%
  pull(key) -> list_cols_numeric

ABC %>% 
  summarise_all(function(x) length(unique(x))) %>%
  gather() %>%
  filter(value > 1) %>%
  pull(key) -> list_cols_morethanone

ABC %>% 
  summarise_all(function(x) all(x>=0)) %>%
  gather() %>%
  filter(value) %>%
  pull(key) -> list_cols_nonnegative


list_cols = intersect(list_cols_numeric, intersect( list_cols_morethanone, list_cols_nonnegative))
testABC <- ABC[,list_cols]
testABC <- cbind.data.frame(case=ABC$case, testABC)


testABC %>% 
  select(list_cols) %>%
  summarise_all(funs(wilcox.test(.~case, data=testABC)$p.value)) -> results

results_df = cbind.data.frame(name=list_cols, result=t(results))
print(results_df[order(results_df$result),])
#                                          result
#y_pred                              2.928116e-51
#ABC.Score                           5.907682e-55
#distance                            9.651027e-79
#normalized_h3K27ac                  1.020357e-05
#distance.1                          9.651027e-79
#hic_contact                         2.992318e-88
#mean.contact.to.TSS                 1.704511e-03
#TSS.count.near.enhancer             2.204992e-04
#mean.contact.from.enhancer          1.174916e-12
#remaining.TSS.contact.from.enhancer 1.094649e-03
#CBFA2T3_e                           1.056667e-04
#POLR2A_e                            3.005241e-03
#TCF12_e                             1.266138e-04
#ARHGAP35_TSS                        2.417411e-03
#C11orf30_TSS                        4.707802e-04
#HDAC6_TSS                           1.867123e-06
#NFXL1_TSS                           1.655301e-03
#RUNX1_TSS                           3.546488e-04
#TAF15_TSS                           5.084246e-05
#UBTF_TSS                            2.744591e-04
#WHSC1_TSS                           3.802608e-03
#XRCC3_TSS                           1.893900e-03
#ZBTB11_TSS                          5.336100e-04
#ZNF639_TSS                          3.792849e-04
#TF_NMF7_e                           4.489936e-04
#TF_NMF11_e                          4.262571e-03
#TF_NMF1_TSS                         1.371376e-03
#TF_NMF4_TSS                         1.010176e-03
#TF_NMF6_TSS                         7.141031e-04
#TF_NMF10_TSS                        2.789534e-03
#nearby.counts                       4.184347e-03
#confusion                           2.928116e-51


sig_cols <- results_df[results < 0.05,1]

for (col in sig_cols) {
  pdf(paste0("weakpos/weak_pos.",col,".wilcox.pdf"), width=3, height=7)
  #print(col)
  p <- ggplot(testABC, aes(x=case, y=.data[[col]]) ) +
    geom_boxplot(position = position_dodge2(preserve = "single")) +
    stat_compare_means(comparisons = list(c("strong pos", "weak pos")), method="wilcox") +
    theme_minimal(base_size = 14)
  print(p)
  dev.off()
}
