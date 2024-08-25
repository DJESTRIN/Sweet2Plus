library(ggplot2)
library(tidyr)
DF<-read.csv("C:/Users/listo/twophoton/summary_data/correlation_by_group.csv")
DF$subject<-as.factor(DF$subject)
DF$cage<-as.factor(DF$cage)
DF$session<-as.factor(DF$session)
DF$neuron<-as.factor(DF$neuron)
DF$uid<-as.factor(paste(DF$cage,DF$subject))
DF$nuid<-as.factor(paste(DF$uid,DF$neuron))


data_long <- gather(DF, period, average_pearson, baseline:posttmt)
data_long$period<-as.factor(data_long$period)
data_long$classification<-as.factor(data_long$classification)
data_long$period <- factor(data_long$period, levels = c("baseline","reward","tmt","posttmt"))
data_long$neurper<-paste(data_long$neuron,data_long$period)
data_long$sessionper<-as.factor(paste(data_long$session,data_long$period))
data_long$nuid<-factor(paste(data_long$subject,data_long$cage,data_long$session,data_long$neuron))
data_long$nuidper<-factor(paste(data_long$nuid,data_long$period))
data_long$session<-factor(data_long$session, c('Day1','Day7','Day14','Day30','Day37'))

#Aggregated dataset
dfav<-aggregate(average_pearson~uid+session+period+sessionper+classification,data=data_long,FUN='mean')
dfav<-aggregate(average_pearson~uid+session+classification,data=dfav,FUN='mean')
dffav<-aggregate(average_pearson~session+classification,data=data_long,FUN='mean')
dffaver<-aggregate(average_pearson~session+classification,data=data_long,FUN=sterr)
dffav$error<-dffaver$average_pearson
  
dffav$session<-factor(dfav$session, c('Day1','Day7','Day14','Day30','Day37'))
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
p<-ggplot(data=dffav,aes(x=session,y=average_pearson))+
  geom_point(aes(group=classification,color=classification,fill=classification))+
  geom_errorbar(aes(ymax=average_pearson+error,ymin=average_pearson-error,group=classification,color=classification,fill=classification))+
  xlab('Session')+
  ylab('Average Pearson Correlation of Neuronal Activity')+
  #scale_x_discrete(labels=c("1"="Stress Naive","7"="7th day of CORT","14"="14th day of CORT","30"="30th day of CORT","37"="2 week recovery"))+
  theme_classic()
print(p)

#Plot across sessions
p<-ggplot(data=dffav,aes(x=classification,y=average_pearson))+
  geom_jitter(data=data_long,aes(x=classification,y=average_pearson,group=nuid),alpha=0.1,width=0.1)+
  geom_point(aes(group=classification,color=classification,fill=classification))+
  geom_errorbar(aes(ymax=average_pearson+error,ymin=average_pearson-error,group=classification,color=classification,fill=classification))+
  xlab('Session')+
  ylab('Average Pearson Correlation of Neuronal Activity')+
  scale_color_manual(values=c("#56B4E9","red"))+
  #scale_x_discrete(labels=c("1"="Stress Naive","7"="7th day of CORT","14"="14th day of CORT","30"="30th day of CORT","37"="2 week recovery"))+
  theme_classic()+
  facet_grid(.~session)
print(p)



dfsess<-aggregate(average_pearson~uid+session,data=dfav,FUN='mean')
p<-ggplot(data=dfsess,aes(x=session,y=average_pearson))+
  geom_boxplot(data=dfsess,aes(x=session,y=average_pearson,color=session),outlier.shape=NA,width=0.5)+
  geom_point(data=dfsess,size=3,alpha=0.5,aes(x=session,y=average_pearson,group=uid),pch = 21)+
  xlab('Session')+
  ylab('Average Pearson Correlation of Neuronal Activity')+
  scale_x_discrete(labels=c("1"="Stress Naive","7"="7th day of CORT","14"="14th day of CORT","30"="30th day of CORT","37"="2 week recovery"))+
  theme_classic()
print(p)

p<-ggplot(data=data_long,aes(x=session,y=average_pearson))+
  geom_jitter(data=data_long,size=3,alpha=0.05,aes(x=session,y=average_pearson,group=nuid),pch = 21)+
  geom_violin(data=data_long,aes(x=session,y=average_pearson,color=session),draw_quantiles = c(0.25, 0.5, 0.75))+
  #geom_jitter(data=data_long,size=3,alpha=0.05,aes(x=session,y=average_pearson,group=nuid),pch = 21)+
  xlab('Session')+
  ylab('Average Pearson Correlation of Neuronal Activity')+
  scale_x_discrete(labels=c("1"="Stress Naive","7"="7th day of CORT","14"="14th day of CORT","30"="30th day of CORT","37"="2 week recovery"))+
  theme_classic()
print(p)

ms <- lmer(average_pearson ~ session + (1 | uid/nuid) ,data=data_long)
anova(ms)
emmeans(ms, list(pairwise ~ session), adjust = "tukey")


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








DF<-read.csv("C:/Users/listo/twophoton/summary_data/correlation_by_group_sep.csv")
DF$subject<-as.factor(DF$subject)
DF$cage<-as.factor(DF$cage)
DF$session<-as.factor(DF$session)
DF$neuron<-as.factor(DF$neuron)
DF$label<-as.factor(DF$label)
DF$uid<-as.factor(paste(DF$cage,DF$subject))
DF$nuid<-as.factor(paste(DF$uid,DF$neuron))


data_long <- gather(DF, period, average_pearson, baseline:posttmt)
data_long$period<-as.factor(data_long$period)
data_long$classification<-as.factor(data_long$label)
data_long$period <- factor(data_long$period, levels = c("baseline","reward","tmt","posttmt"))
data_long$neurper<-paste(data_long$neuron,data_long$period)
data_long$sessionper<-as.factor(paste(data_long$session,data_long$period))
data_long$nuid<-factor(paste(data_long$subject,data_long$cage,data_long$session,data_long$neuron))
data_long$nuidper<-factor(paste(data_long$nuid,data_long$period))
data_long$session<-factor(data_long$session, c('Day1','Day7','Day14','Day30','Day37'))

#Aggregated dataset
dfav<-aggregate(average_pearson~uid+session+period+sessionper+classification,data=data_long,FUN='mean')
dfav<-aggregate(average_pearson~uid+session+classification,data=dfav,FUN='mean')
dffav<-aggregate(average_pearson~session+classification,data=data_long,FUN='mean')
dffaver<-aggregate(average_pearson~session+classification,data=data_long,FUN=sterr)
dffav$error<-dffaver$average_pearson

#Plot across sessions
p<-ggplot(data=dffav,aes(x=classification,y=average_pearson))+
  geom_jitter(data=data_long,aes(x=classification,y=average_pearson,group=nuid),alpha=0.1,width=0.1)+
  geom_point(aes(group=classification,color=classification,fill=classification))+
  geom_errorbar(aes(ymax=average_pearson+error,ymin=average_pearson-error,group=classification,color=classification,fill=classification))+
  xlab('Session')+
  ylab('Average Pearson Correlation of Neuronal Activity')+
  scale_color_manual(values=c("#56B4E9","red"))+
  #scale_x_discrete(labels=c("1"="Stress Naive","7"="7th day of CORT","14"="14th day of CORT","30"="30th day of CORT","37"="2 week recovery"))+
  theme_classic()+
  facet_grid(.~session)
print(p)





m1 <- lme(average_pearson~session*classification, random=~1|(uid/nuid),data=data_long,na.action=na.omit)
anova(m1)
emmeans(m1, list(pairwise ~ session*classification), adjust = "bonferroni")





