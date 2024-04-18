library(dplyr)
library(ggplot2)

theme_set(theme_minimal(base_size = 14))


plotbox <- function(dataname, outdir) {

  testABC <- read.table(paste0(dataname,".ABC.confusion.features.txt"), header=TRUE, sep="\t")
  test <- testABC
  test$nearby.counts = test$Enhancer.count.near.TSS + test$TSS.count.near.enhancer
  test$confusion = as.factor(paste(test$Significant, test$y_pred, sep=""))
  test <- test %>% mutate(enhancer_cnt = cut(Enhancer.count.near.TSS, breaks=c(0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000)), .drop = FALSE)
  test <- test %>% mutate(TSS_cnt = cut(TSS.count.near.enhancer, breaks=c(0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220)), .drop = FALSE)
  test <- test %>% mutate(nearby_cnt = cut(nearby.counts, breaks=c(0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000)), .drop = FALSE)

  testcnt <- test %>% count(TSS_cnt, confusion, .drop = FALSE) 
  pdf(paste0(outdir,"/ABC.confusion_by_cnt.boxplot.pdf"))
  p<-ggplot(testcnt, aes(x=TSS_cnt, y=n, fill=confusion)) +
    geom_bar(stat="identity", color="black", position=position_dodge()) +
    scale_x_discrete(labels=c(10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210)) +
    ylim(0, 12000) +
    theme_minimal(base_size = 14)
  print(p)
  testcnt <- testcnt[testcnt$confusion != "00",]
  p<-ggplot(testcnt, aes(x=TSS_cnt, y=n, fill=confusion)) +
    geom_bar(stat="identity", color="black", position=position_dodge())+
    scale_x_discrete(labels=c(10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210)) +
    ylim(0, 2500) +
    theme_minimal(base_size = 14)
  print(p)
  p<-ggplot(testcnt, aes(x=TSS_cnt, y=n, fill=confusion)) +
    scale_x_discrete(labels=c(10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210)) +
    geom_col(position = "fill")+
    theme_minimal(base_size = 14)
  print(p)
  dev.off()

  pdf(paste0(outdir,"/ABC.hic_confusion_by_cnt.boxplot.pdf"))
  p <- ggplot(test, aes(x=enhancer_cnt, y=hic_contact, fill=confusion) ) + 
    geom_boxplot(position = position_dodge2(preserve = "single")) +
    scale_x_discrete(labels=c(100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900)) +
    theme_minimal(base_size = 14)
  print(p)
  p <- ggplot(test, aes(x=TSS_cnt, y=hic_contact, fill=confusion) ) + 
    geom_boxplot(position = position_dodge2(preserve = "single")) +
    scale_x_discrete(labels=c(10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210)) +
    theme_minimal(base_size = 14)
  print(p)
  dev.off()


  pdf(paste0(outdir,"/hic_significance.boxplot.pdf"))
  test$Significant = as.factor(test$Significant)
  p <- ggplot(test, aes(x=Significant, y=hic_contact, fill=Significant) ) + 
    geom_boxplot(position = position_dodge2(preserve = "single")) +
    scale_x_discrete(labels=c(100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900)) +
    theme_minimal(base_size = 14)
  print(p)
  test$strongcontact = ifelse(test$hic_contact >= 0.005, "strong", "weak")
  test$siglabel = ifelse(test$Significant=="1", "pos", "neg")
  test$case = as.factor(paste(test$strongcontact, test$siglabel))
  testcnt <- test %>% count(case, .drop = FALSE) 
  p<-ggplot(testcnt, aes(x=case, y=n, fill=case)) +
    geom_bar(stat="identity", color="black", position=position_dodge()) +
    theme_minimal(base_size = 14)
  print(p)
  p<-ggplot(testcnt, aes(x=1, y=n, fill=case)) +
    geom_col(position = "fill") +
    coord_flip() + 
    theme_minimal(base_size = 14)
  print(p)
  testcnt <- testcnt[testcnt$case != "weak neg",]
  p<-ggplot(testcnt, aes(x=case, y=n, fill=case)) +
    geom_bar(stat="identity", color="black", position=position_dodge())+
    theme_minimal(base_size = 14)
  print(p)
  dev.off()

  # save positives
  #write.table(test[test$siglabel == "pos",], "pos.gasperini.txt", sep="\t", quote=FALSE)
  #write.table(test[test$case == "weak pos",], "weak_pos.gasperini.txt", sep="\t", quote=FALSE)

  pdf(paste0(outdir,"/hic_significance_by_cnt.boxplot.pdf"))
  test$Significant = as.factor(test$Significant)
  p <- ggplot(test, aes(x=enhancer_cnt, y=hic_contact, fill=Significant) ) + 
    geom_boxplot(position = position_dodge2(preserve = "single")) +
    scale_x_discrete(labels=c(100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900)) +
    theme_minimal(base_size = 14)
  print(p)
  p <- ggplot(test, aes(x=TSS_cnt, y=hic_contact, fill=Significant) ) + 
    geom_boxplot(position = position_dodge2(preserve = "single")) +
    scale_x_discrete(labels=c(10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210)) +
    theme_minimal(base_size = 14)
  print(p)
  test$strongcontact = ifelse(test$hic_contact >= 0.005, "strong", "weak")
  test$siglabel = ifelse(test$Significant=="1", "pos", "neg")
  test$case = paste(test$strongcontact, test$siglabel)
  testcnt <- test %>% count(nearby_cnt, case, .drop = FALSE) 
  p<-ggplot(testcnt, aes(x=nearby_cnt, y=n, fill=case)) +
    geom_bar(stat="identity", color="black", position=position_dodge()) +
    scale_x_discrete(labels=c(100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900)) +
    theme_minimal(base_size = 14)
  print(p)
  testcnt <- testcnt[testcnt$case != "weak neg",]
  p<-ggplot(testcnt, aes(x=nearby_cnt, y=n, fill=case)) +
    geom_bar(stat="identity", color="black", position=position_dodge())+
    scale_x_discrete(labels=c(100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900)) +
    theme_minimal(base_size = 14)
  print(p)
  p<-ggplot(testcnt, aes(x=nearby_cnt, y=n, fill=case)) +
    scale_x_discrete(labels=c(100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900)) +
    geom_col(position = "fill")+
    theme_minimal(base_size = 14)
  print(p)

  dev.off()




  pdf(paste0(outdir,"/hic_significance_by_chr.boxplot.pdf"))
  test$chr = unlist(strsplit(test$ABC.id, ":"))[c(TRUE,FALSE)]
  test$chr = ordered(test$chr, levels = c('chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'))

  p <- ggplot(test, aes(x=chr, y=hic_contact, fill=Significant) ) + 
    geom_boxplot(position = position_dodge2(preserve = "single")) +
    scale_x_discrete(labels=c(1:8, 10:22, 'X'))  +
    theme_minimal(base_size = 14)
  print(p)
  test$strongcontact = ifelse(test$hic_contact >= 0.005, "strong", "weak")
  test$siglabel = ifelse(test$Significant=="1", "pos", "neg")
  test$case = paste(test$strongcontact, test$siglabel)
  testcnt <- test %>% count(chr, case, .drop = FALSE) 
  p<-ggplot(testcnt, aes(x=chr, y=n, fill=case)) +
    geom_bar(stat="identity", color="black", position=position_dodge()) +
    scale_x_discrete(labels=c(1:8, 10:22, 'X')) +
    theme_minimal(base_size = 14)
  print(p)
  testcnt <- testcnt[testcnt$case != "weak neg",]
  p<-ggplot(testcnt, aes(x=chr, y=n, fill=case)) +
    geom_bar(stat="identity", color="black", position=position_dodge())+
    scale_x_discrete(labels=c(1:8, 10:22, 'X')) +
    theme_minimal(base_size = 14)
  print(p)
  p<-ggplot(testcnt, aes(x=chr, y=n, fill=case)) +
    scale_x_discrete(labels=c(1:8, 10:22, 'X')) +
    geom_col(position = "fill") +
    theme_minimal(base_size = 14)
  print(p)

  dev.off()






  # data
  pdf(paste0(outdir,"/ABC_by_cnt.boxplot.pdf"))
  p <- ggplot(test, aes(x=nearby.counts, y=ABC.Score)) + geom_point() + ylim(0, 0.15) + geom_smooth() +
    theme_minimal(base_size = 14)
  print(p)
  p <- ggplot(test, aes(x=nearby.counts, y=hic_contact)) + geom_point() + geom_smooth() +
    theme_minimal(base_size = 14)
  p 
  p <- ggplot(test, aes(x=TSS_cnt, y=hic_contact) ) + 
    geom_boxplot(position = position_dodge2(preserve = "single")) + ylim(0, 0.003) + stat_summary(fun.y="mean", ,color="red") +
    scale_x_discrete(labels=c(10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210)) +
    theme_minimal(base_size = 14)
  print(p)
  #p <- ggplot(test, aes(x=nearby.counts, y=mean.contact.to.TSS)) + geom_point() + geom_smooth() +
  #  theme_minimal(base_size = 14)
  #print(p)
  #p <- ggplot(test, aes(x=nearby.counts, y=mean.contact.from.enhancer)) + geom_point() + geom_smooth() +
  #  theme_minimal(base_size = 14)
  #print(p)
  dev.off()

  pdf(paste0(outdir,"/remaining_vs_count.pdf"))
  enhancer.lm <- lm(remaining.enhancers.contact.to.TSS ~ Enhancer.count.near.TSS, test)
  p <- ggplot(test, aes(y=remaining.enhancers.contact.to.TSS, x=Enhancer.count.near.TSS)) + 
    geom_point(alpha = 0.1) + 
    geom_abline(slope = coef(enhancer.lm)[["Enhancer.count.near.TSS"]], 
              intercept = coef(enhancer.lm)[["(Intercept)"]],
              colour='#E41A1C')
  print(p)
  TSS.lm <- lm(remaining.TSS.contact.from.enhancer ~ TSS.count.near.enhancer, test)
  p <- ggplot(test, aes(y=remaining.TSS.contact.from.enhancer, x=TSS.count.near.enhancer)) + 
    geom_point(alpha = 0.1) + 
    geom_abline(slope = coef(TSS.lm)[["TSS.count.near.enhancer"]], 
              intercept = coef(TSS.lm)[["(Intercept)"]], 
              colour='#E41A1C')
  print(p)
  dev.off()




  # XGB
  testXGB <- read.table(paste0(dataname,".XGB.confusion.features.txt"), header=TRUE, sep="\t")
  test <- testXGB
  test$y_pred = as.integer(as.logical(test$y_pred))
  test$nearby.counts = test$Enhancer.count.near.TSS + test$TSS.count.near.enhancer
  test$confusion = as.factor(paste(test$Significant, test$y_pred, sep=""))
  test <- test %>% mutate(enhancer_cnt = cut(Enhancer.count.near.TSS, breaks=c(0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000)), .drop = FALSE)
  test <- test %>% mutate(TSS_cnt = cut(TSS.count.near.enhancer, breaks=c(0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220)), .drop = FALSE)

  testcnt <- test %>% count(TSS_cnt, confusion, .drop = FALSE)
  pdf(paste0(outdir,"/XGB.confusion_by_cnt.boxplot.pdf"))
  p<-ggplot(testcnt, aes(x=TSS_cnt, y=n, fill=confusion)) +
    geom_bar(stat="identity", color="black", position=position_dodge())+
    scale_x_discrete(labels=c(10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210)) +
    ylim(0, 12000) +
    theme_minimal(base_size = 14)
  print(p)
  testcnt <- testcnt[testcnt$confusion != "00",]
  p<-ggplot(testcnt, aes(x=TSS_cnt, y=n, fill=confusion)) +
    geom_bar(stat="identity", color="black", position=position_dodge())+
    scale_x_discrete(labels=c(10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210)) +
    ylim(0, 2500) +
    theme_minimal(base_size = 14)
  print(p)
  p<-ggplot(testcnt, aes(x=TSS_cnt, y=n, fill=confusion)) +
    geom_col(position = "fill") +
    scale_x_discrete(labels=c(10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210)) +
    theme_minimal(base_size = 14)
  print(p)
  dev.off()


  pdf(paste0(outdir,"/XGB.hic_confusion_by_cnt.boxplot.pdf"))
  p <- ggplot(test, aes(x=enhancer_cnt, y=hic_contact, fill=confusion) ) + 
    geom_boxplot(position = position_dodge2(preserve = "single")) +
    scale_x_discrete(labels=c(100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900))  +
    theme_minimal(base_size = 14)
  print(p)
  p <- ggplot(test, aes(x=TSS_cnt, y=hic_contact, fill=confusion) ) + 
    geom_boxplot(position = position_dodge2(preserve = "single")) +
    scale_x_discrete(labels=c(10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210)) +
    theme_minimal(base_size = 14)
  print(p)
  dev.off()


}

plotbox("gasperini.all", "gasperini.boxplot")
plotbox("fulco", "fulco.boxplot")
plotbox("shraivogel", "shraivogel.boxplot")




