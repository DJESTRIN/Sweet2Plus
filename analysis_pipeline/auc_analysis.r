library(ggplot2)
library(tidyr)
DF<-read.csv("C:/Users/listo/twophoton/summary_data/auc_tall.csv")
DF$subject<-as.factor(DF$subject)
DF$cage<-as.factor(DF$cage)
DF$session<-as.factor(DF$session)
DF$neuron<-as.factor(DF$neuron)
DF$label<-as.factor(DF$label)
DF$uid<-as.factor(paste(DF$cage,DF$subject))
DF$session<-factor(DF$session,c('Day1','Day7','Day14','Day30','Day37'))
DF$nuid<-as.factor(paste(DF$uid,DF$neuron))

data_long <- gather(DF, trialtype, auc, water:tmt)
data_long$trialtype<-factor(data_long$trialtype,c("water","vanilla","peanutbutter","tmt"))

# all changes in auc for all neuron types:
dataav<-aggregate(auc~trialtype+uid+session,data=data_long,FUN='mean')
averages<-aggregate(auc~trialtype+session,data=dataav,FUN='mean')
errors<-aggregate(auc~trialtype+session,data=dataav,FUN=sterr)
averages$error<-errors$auc

p<-ggplot(data=data_long,aes(x=trialtype,y=auc))+
  geom_jitter(alpha=0.1)+
  geom_point(data=averages,aes(x=trialtype,y=auc,color=trialtype,fill=trialtype))+
  geom_errorbar(data=averages,aes(x=trialtype,y=auc,ymin=auc-error,ymax=auc+error,color=trialtype))+
  facet_grid(~session)+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
print(p)

m1 <- lme(auc~session*trialtype ,random=~1|uid/nuid,data=data_long)
anova(m1)
emmeans(m1, list(pairwise ~ session*trialtype), adjust = "fdr")


# changes in auc across neuron groups
dataav<-aggregate(auc~trialtype+uid+session+label,data=data_long,FUN='mean')
averages<-aggregate(auc~trialtype+session+label,data=dataav,FUN='mean')
errors<-aggregate(auc~trialtype+session+label,data=dataav,FUN=sterr)
averages$error<-errors$auc

#plot general auc data
p<-ggplot(data=data_long,aes(x=trialtype,y=auc))+
  geom_jitter()+
  geom_point(data=averages,aes(x=trialtype,y=auc,color=trialtype,fill=trialtype))+
  geom_errorbar(data=averages,aes(x=trialtype,y=auc,ymin=auc-error,ymax=auc+error,color=trialtype))+
  facet_grid(label~session)+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
print(p)