
library(knitr)
library(kableExtra)
library(ggplot2)
library(tidyr)
library(dplyr)
library(lme4)
library(lmerTest)
library(grid)

# Custom functions
standard_error <- function(x) {
  return(sd(x) / sqrt(length(x)))}

# Load data
DF<-read.csv("C:/Users/listo/Sweet2Plus/Repeated_Measures_Correlations.csv")

# Clean data
DF$subject<-as.factor(DF$subject)
DF$cage<-as.factor(DF$cage)
DF$session<-as.factor(DF$session)
DF$group<-as.factor(DF$group)
DF$neuron<-as.factor(DF$neuron)
DF$uid<-as.factor(paste(DF$cage,DF$subject))
DF$classification<-as.factor(DF$classification)

# Convert data from wide to long format
data_long <- gather(DF, period, average_pearson, baseline:posttmt)
data_long$period<-as.factor(data_long$period)
data_long$period <- factor(data_long$period, levels = c("baseline","reward","tmt","posttmt"))
data_long$nuid <-as.factor(paste(data_long$cage,data_long$subject,data_long$neuron))

bl<-data_long[data_long$period=="baseline",]
p<-ggplot(data=bl,aes(x=session,y=average_pearson,color=classification,group=nuid))+
  geom_line()+
  facet_grid(.~group)
print(p)

bl <- bl %>%
  group_by(nuid) %>%
  mutate(
    mean_avs = average_pearson[session == 0][1],  # Calculate the mean avs for each nuid
    sd_avs = sd(average_pearson),      # Calculate the standard deviation of avs for each nuid
    normalized_pearson = (average_pearson - mean_avs) / sd_avs  # Calculate the Z-score
  ) %>%
  select(-mean_avs, -sd_avs) %>% # Divide by avs at session 0
  ungroup()

bl_intermediate_av_neu<-aggregate(normalized_pearson~group+session+uid+nuid,data=bl,FUN='mean')
bl_intermediate_av<-aggregate(normalized_pearson~group+session+uid,data=bl_intermediate_av_neu,FUN='mean')
bl_av<-aggregate(normalized_pearson~group+session,data=bl_intermediate_av,FUN='mean')
bl_errs<-aggregate(normalized_pearson~group+session,data=bl_intermediate_av,FUN=standard_error)
bl_av$errs<-bl_errs$normalized_pearson

# Generate lmer and anova stats
pdf("output_with_ggplot1.pdf", width = 8, height = 6)  # Customize size
baseline_correlation_model <- lmer(normalized_pearson ~ session * group + (1 | uid/nuid) ,data=bl_intermediate_av_neu)
print(summary(baseline_correlation_model))

grid.newpage()
res<-anova(baseline_correlation_model)
kr<-kable(res, caption = "res") %>%
  kable_styling(latex_options = c("striped", "hold_position"))
print(kr)


plot(baseline_correlation_model, which = 1, col = "blue", pch = 20)

p<-ggplot(data=bl_av,aes(x=session,y=normalized_pearson))+
  geom_line(data=bl_intermediate_av_neu,aes(x=session,y=normalized_pearson,group=nuid),alpha=0.01)+
  geom_errorbar(aes(ymin=normalized_pearson-errs,ymax=normalized_pearson+errs))+
  geom_point()+
  facet_grid(.~group) +
  labs(y = expression(
    "Normalized Pearson Correlation (PC)" == frac(
      "PC"[t] - "PC"[t == 0],
      "Standard Deviation PC"[t == 0:30])))+
  theme(axis.title.y = element_text(size = 5))+
  theme_minimal()
print(p)
grid.newpage()
dev.off()
# ggsave("C:/Users/listo/Sweet2Plus/my_figs/normalized_pearson_correlation_baseline_nuid.jpg", dpi = 300, width = 6, height = 8)

# baseline_correlation_model <- lmer(normalized_pearson ~ session * group + (1 | uid/nuid) ,data=bl_intermediate_av_neu)
# browser()
# anova(baseline_correlation_model)
#m1 <- lme(average_pearson~session*period ,random=~1|uid/nuid,data=data_long)
# anova(m1)
# #lsmeans(m1, pairwise~session*period, adjust="fdr")
# emmeans(fm, list(pairwise ~ session*period), adjust = "tukey")

# browser()
# #Plot Neuronal Activity across Sessions, trial types and groups
# p<-ggplot(data=data_long,aes(x=session,y=average_pearson,color=))+
#   #geom_violin(data=data_long,aes(x=session,y=average_pearson),alpha=0.1)+
#   geom_boxplot(data=dfav,aes(x=session,y=average_pearson,group=sessionper,color=period),outlier.shape=NA,width=0.5)+
#   #geom_jitter(data=dfav,aes(x=session,y=average_pearson,group=sessionper,color=period),size=2.5,position = position_dodge())+
#   geom_point(data=dfav,size=3,alpha=0.5,aes(x=session,y=average_pearson,group=sessionper,color=period,fill=period),pch = 21, 
#              position = position_jitterdodge(dodge.width=0.5))+
#   xlab('Session')+
#   ylab('Average Pearson Correlation of Neuronal Activity')+
#   scale_x_discrete(labels=c("1"="Stress Naive","7"="7th day of CORT","14"="14th day of CORT","30"="30th day of CORT","37"="2 week recovery"))+
#   theme_classic()
# print(p)

# #Aggregated dataset
# dfav<-aggregate(average_pearson~uid+session+period+sessionper,data=data_long,FUN='mean')
# dfav$period <- factor(dfav$period, levels = c("baseline","reward","tmt","posttmt"))
# dfav$sessionper <- factor(dfav$sessionper, levels = c("1 baseline","1 reward","1 tmt","1 posttmt",
#                                                       "7 baseline","7 reward","7 tmt","7 posttmt",
#                                                       "14 baseline","14 reward","14 tmt","14 posttmt"))

# fm <- lmer(average_pearson ~ session * period + (1 | uid/nuid) ,data=data_long)
# m1 <- lme(average_pearson~session*period ,random=~1|uid/nuid,data=data_long)
# anova(m1)
# #lsmeans(m1, pairwise~session*period, adjust="fdr")
# emmeans(fm, list(pairwise ~ session*period), adjust = "tukey")

# #Plot across sessions
# p<-ggplot(data=dfav,aes(x=session,y=average_pearson))+
#   #geom_violin(data=data_long,aes(x=session,y=average_pearson),alpha=0.1)+
#   geom_boxplot(data=dfav,aes(x=session,y=average_pearson,group=sessionper,color=period),outlier.shape=NA,width=0.5)+
#   #geom_jitter(data=dfav,aes(x=session,y=average_pearson,group=sessionper,color=period),size=2.5,position = position_dodge())+
#   geom_point(data=dfav,size=3,alpha=0.5,aes(x=session,y=average_pearson,group=sessionper,color=period,fill=period),pch = 21, 
#              position = position_jitterdodge(dodge.width=0.5))+
#   xlab('Session')+
#   ylab('Average Pearson Correlation of Neuronal Activity')+
#   scale_x_discrete(labels=c("1"="Stress Naive","7"="7th day of CORT","14"="14th day of CORT","30"="30th day of CORT","37"="2 week recovery"))+
#   theme_classic()
# print(p)


# dfsess<-aggregate(average_pearson~uid+session,data=dfav,FUN='mean')
# p<-ggplot(data=dfsess,aes(x=session,y=average_pearson))+
#   geom_boxplot(data=dfsess,aes(x=session,y=average_pearson,color=session),outlier.shape=NA,width=0.5)+
#   geom_point(data=dfsess,size=3,alpha=0.5,aes(x=session,y=average_pearson,group=uid),pch = 21)+
#   xlab('Session')+
#   ylab('Average Pearson Correlation of Neuronal Activity')+
#   scale_x_discrete(labels=c("1"="Stress Naive","7"="7th day of CORT","14"="14th day of CORT","30"="30th day of CORT","37"="2 week recovery"))+
#   theme_classic()
# print(p)

# p<-ggplot(data=data_long,aes(x=session,y=average_pearson))+
#   geom_jitter(data=data_long,size=3,alpha=0.05,aes(x=session,y=average_pearson,group=nuid),pch = 21)+
#   geom_violin(data=data_long,aes(x=session,y=average_pearson,color=session),draw_quantiles = c(0.25, 0.5, 0.75))+
#   #geom_jitter(data=data_long,size=3,alpha=0.05,aes(x=session,y=average_pearson,group=nuid),pch = 21)+
#   xlab('Session')+
#   ylab('Average Pearson Correlation of Neuronal Activity')+
#   scale_x_discrete(labels=c("1"="Stress Naive","7"="7th day of CORT","14"="14th day of CORT","30"="30th day of CORT","37"="2 week recovery"))+
#   theme_classic()
# print(p)

# ms <- lmer(average_pearson ~ session + (1 | uid/nuid) ,data=data_long)
# anova(ms)
# emmeans(ms, list(pairwise ~ session), adjust = "tukey")


# res.aov <- aov(average_pearson ~ session*period + Error(uid), data = dfav)
# res.aov <- anova_test(data = dfav, dv = average_pearson, wid = uid, within = c(session, period))
# get_anova_table(res.aov)

# pwc <- dfav %>%
#   pairwise_t_test(
#     average_pearson ~ period,
#     p.adjust.method = "bonferroni"
#   )

# # Summary of the analysis
# summary(res.aov)