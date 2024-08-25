library(ggplot2)
library(tidyr)
library(lme4)
library(nlme)
library(emmeans)
library(dplyr)

DF<-read.csv("C:/Users/listo/twophoton/summary_data/auc_tall.csv")
DF$subject<-as.factor(DF$subject)
DF$cage<-as.factor(DF$cage)
DF$session<-as.factor(DF$session)
DF$neuron<-as.factor(DF$neuron)
DF$label<-as.factor(DF$label)
DF$uid<-as.factor(paste(DF$cage,DF$subject))
DF$session<-factor(DF$session,c('Day1','Day7','Day14','Day30','Day37'))
DF$nuid<-as.factor(paste(DF$uid,DF$neuron))


DFmax<-DF[,c("water","vanilla","peanutbutter","tmt")]  
DFmax <- DFmax %>% rowwise %>% mutate(Max = names(.)[which.max(c(water, vanilla, peanutbutter,tmt))]) %>% ungroup
DF$max_auc<-DFmax$Max
response_pie_data<-DF %>% count(max_auc, session)
response_pie_data<-response_pie_data %>% group_by(session) %>%
  mutate(per =  100 *n/sum(n)) %>% ungroup

# Basic piechart
p<-ggplot(response_pie_data, aes(x="", y=per, fill=max_auc)) +
  geom_bar(stat="identity", width=1) +
  coord_polar("y", start=0)+
  facet_grid(~session)
print(p)

response_pie_data_an<-DF %>% count(max_auc, session,uid)
response_pie_data_an<-response_pie_data_an %>% group_by(session,uid) %>%
  mutate(per =  100 *n/sum(n)) %>% ungroup
response_pie_data_an$grouping<-as.factor(paste(response_pie_data_an$max_auc,response_pie_data_an$uid))

averages<-aggregate(data=response_pie_data_an,per~session+max_auc,FUN='mean')
errs<-aggregate(data=response_pie_data_an,per~session+max_auc,FUN=sterr)
averages$error<-errs$per

p<-ggplot(response_pie_data_an, aes(x=session, y=per, color=max_auc,group=grouping)) +
  #geom_line(alpha=0.5,size=1) +
  geom_ribbon(data=averages,aes(x=session, y=per, ymax=per+error, 
                                ymin=per-error,color=max_auc,fill=max_auc,group=max_auc),alpha=0.5,size=2)+
  geom_line(data=averages,aes(x=session, y=per,color=max_auc,group=max_auc),size=2)
print(p)

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
  geom_hline(yintercept=0, linetype='dashed', col = 'red',size=1)+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
print(p)

p<-ggplot(data=data_long,aes(fill=trialtype,y=auc))+
  geom_density(alpha=0.4)+
  facet_grid(~session)
print(p)


m1 <- lme(auc~session*trialtype ,random=~1|uid/nuid,data=data_long)
anova(m1)
emmeans(m1, list(pairwise ~ session*trialtype), adjust = "bonferroni")
ncvTest(m1)

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
  geom_hline(yintercept=0, linetype='dashed', col = 'red')+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
print(p)

m1 <- lme(auc~session*trialtype*label ,random=~1|uid/nuid,data=data_long)
anova(m1)
emmeans(m1, list(pairwise ~ session*trialtype*label), adjust = "bonferroni")