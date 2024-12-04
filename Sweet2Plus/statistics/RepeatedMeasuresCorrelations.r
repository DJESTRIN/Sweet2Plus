
library(knitr)
library(kableExtra)
library(ggplot2)
library(tidyr)
library(dplyr)
library(lme4)
library(lmerTest)
library(grid)
library(emmeans)

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
baseline_correlation_model <- lmer(normalized_pearson ~ session * group + (1 | uid/nuid) ,data=bl_intermediate_av_neu)
print(summary(baseline_correlation_model))

res<-anova(baseline_correlation_model)
print(res)

ems<-emmeans(baseline_correlation_model, list(pairwise ~ session*group), adjust = "bonferroni")
print(ems)

residuals_norm <- residuals(baseline_correlation_model)
residuals_df <- data.frame(residuals = residuals_norm)

# QQ plot with ggplot2
p<-ggplot(residuals_df, aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line(color = "red") +
  theme_minimal() +
  labs(title = "QQ Plot of Residuals", x = "Theoretical Quantiles", y = "Sample Quantiles")
print(p)
ggsave("C:/Users/listo/Sweet2Plus/my_figs/model_fit.jpg", dpi = 300, width = 6, height = 8)

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

ggsave("C:/Users/listo/Sweet2Plus/my_figs/normalized_pearson_correlation_baseline_nuid.jpg", dpi = 300, width = 6, height = 8)




## How does pearson correlation change across conditions and time
al <- data_long
al <- al %>%
  group_by(nuid,period) %>%
  mutate(
    mean_avs = average_pearson[session == 0][1],  # Calculate the mean avs for each nuid
    sd_avs = sd(average_pearson),      # Calculate the standard deviation of avs for each nuid
    normalized_pearson = (average_pearson - mean_avs) / sd_avs  # Calculate the Z-score
  ) %>%
  select(-mean_avs, -sd_avs) %>% # Divide by avs at session 0
  ungroup()

al$group <- factor(al$group, levels = c("control", "cort"),
  labels = c("Vehicle Control", "CORT"))

al$period <- factor(al$period, levels = c("baseline", "reward", "tmt", "posttmt"),
  labels = c("Pre-Task Baseline","Reward Trials", "Fox Urine Trials","Post-Task Baseline"))

intermediate_av_neu<-aggregate(normalized_pearson~group+session+uid+nuid+period,data=al,FUN='mean')
all_avs<-aggregate(normalized_pearson~group+session+uid+period,data=intermediate_av_neu,FUN='mean')
all_avs_means<-aggregate(normalized_pearson~group+session+period,data=all_avs,FUN='mean')
all_avs_errors<-aggregate(normalized_pearson~group+session+period,data=all_avs,FUN=standard_error)
all_avs_means$errors<-all_avs_errors$normalized_pearson

p<-ggplot(data=all_avs_means,aes(x=session,y=normalized_pearson,color=period,group=period))+
  geom_line(data=intermediate_av_neu,aes(x=session,y=normalized_pearson,group=nuid),alpha=0.01)+
  geom_point()+
  geom_errorbar(aes(ymin=normalized_pearson-errors,ymax=normalized_pearson+errors),width=0)+
  geom_vline(xintercept = 1.5, linetype = "dashed", color = "red") +
  facet_grid(period~group)+
  labs(y = expression(
    "Normalized Pearson Correlation (PC)" == frac(
      "PC"[t] - "PC"[t == 0],
      "Standard Deviation PC"[t == 0:30])))+
  theme(axis.title.y = element_text(size = 5))+
  ggtitle("Normalized Pearson Correlations w.r.t. Group, Trial Type and Session") +
  theme_minimal() +
  theme(legend.position = "none",plot.title = element_text(size = 10))
print(p)
ggsave("C:/Users/listo/Sweet2Plus/my_figs/norm_perasoncorrelations_period_session_group.jpg", dpi = 300, width = 6, height = 8)
