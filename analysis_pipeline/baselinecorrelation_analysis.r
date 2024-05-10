library(ggplot2)
library(tidyr)
DF<-read.csv("C:/Users/listo/twophoton/repeatedmeasures_correlations_all.csv")
DF$subject<-as.factor(DF$subject)
DF$cage<-as.factor(DF$cage)
DF$session<-as.factor(DF$session)
DF$neuron<-as.factor(DF$neuron)
DF$uid<-as.factor(paste(DF$cage,DF$subject))


data_long <- gather(DF, period, average_pearson, baseline:posttmt)
data_long$period<-as.factor(data_long$period)
data_long$period <- factor(data_long$period, levels = c("baseline","reward","tmt","posttmt"))
data_long$neurper<-paste(data_long$neuron,data_long$period)
data_long$sessionper<-as.factor(paste(data_long$session,data_long$period))
data_long$nuid<-factor(paste(data_long$subject,data_long$cage,data_long$session,data_long$neuron))
data_long$nuidper<-factor(paste(data_long$nuid,data_long$period))
#Aggregated dataset
dfav<-aggregate(average_pearson~uid+session+period+sessionper,data=data_long,FUN='mean')
dfav$period <- factor(dfav$period, levels = c("baseline","reward","tmt","posttmt"))
dfav$sessionper <- factor(dfav$sessionper, levels = c("1 baseline","1 reward","1 tmt","1 posttmt",
                                                      "7 baseline","7 reward","7 tmt","7 posttmt",
                                                      "14 baseline","14 reward","14 tmt","14 posttmt"))

fm <- lmer(average_pearson ~ session * period + (1 | uid/nuid) ,data=data_long)
m1 <- lme(average_pearson~session*period ,random=~1|uid/nuid,data=data_long)
anova(m1)
#lsmeans(m1, pairwise~session*period, adjust="fdr")
emmeans(fm, list(pairwise ~ session*period), adjust = "tukey")

#Plot across sessions
p<-ggplot(data=dfav,aes(x=session,y=average_pearson))+
  geom_jitter(data=data_long,aes(x=session,y=average_pearson,group=nuidper),alpha=0.1,width=0.05)+
  geom_boxplot(data=dfav,aes(x=session,y=average_pearson,group=sessionper,color=period),outlier.shape=NA,width=0.5)+
  #geom_jitter(data=dfav,aes(x=session,y=average_pearson,group=sessionper,color=period),size=2.5,position = position_dodge())+
  geom_point(data=dfav,size=3,alpha=0.5,aes(x=session,y=average_pearson,group=sessionper,color=period,fill=period),pch = 21, 
             position = position_jitterdodge(dodge.width=0.5))+
  xlab('Session')+
  ylab('Average Pearson Correlation of Neuronal Activity')+
  theme_classic()
print(p)

res.aov <- aov(average_pearson ~ session*period + Error(uid), data = dfav)
res.aov <- anova_test(data = dfav, dv = average_pearson, wid = uid, within = c(session, period))
get_anova_table(res.aov)

pwc <- dfav %>%
  pairwise_t_test(
    average_pearson ~ period,
    p.adjust.method = "bonferroni"
  )

# Summary of the analysis
summary(res.aov)