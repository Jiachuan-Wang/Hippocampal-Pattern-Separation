#Cosine similarity==============================
library(tidyverse)
library(ggplot2)
library(scales)
library(viridis)
library(latex2exp)

files <- list.files(path = "demo", 
                        pattern = "^250605.*similarity\\.csv$")

df_combined <- map_dfr(files, function(file) {
  # Extract learning rate from filename
  lr_val <- as.numeric(str_match(file, "_([0-9.e-]+)lr")[, 2])
  # Extract condition type
  condition <- str_extract(file, "(mature|young)_\\dadd")
  # Match condition to label
  condition_label <- case_when(
    condition == "mature_0add" ~ "No neurogenesis",
    condition == "mature_1add" ~ "Neurogenesis - Mature",
    condition == "young_1add"  ~ "Neurogenesis - Young",
    TRUE ~ "Unknown"
  )
  # Read and return with metadata
  read_csv(paste0("demo\\", file)) %>%
    mutate(
      lr = lr_val,
      condition = condition,
      condition_label = condition_label,
      file_name = file
    )
})

df_combined$...1 = df_combined$...1 + 1

standard_error <- function(x) {
  sd(x) / sqrt(length(x))
}

df <- df_combined %>%
  group_by(lr) %>%
  mutate(replication = rep(1:(n() / 14), each = 14)) %>%
  ungroup()

rep_means <- df %>%
  group_by(lr, replication, condition_label) %>%
  summarise(mean_cosine = mean(Cosine, na.rm = TRUE), .groups = "drop")

test <- rep_means %>%
  group_by(lr, condition_label) %>%
  summarise(
    Mean = mean(mean_cosine),
    SE = sd(mean_cosine) / sqrt(n()),
    .groups = "drop"
  )

ggplot(test, aes(x = lr, y = Mean, group=condition_label, color=condition_label, linetype = condition_label)) +
  geom_line() +  # Line plot
  geom_ribbon(aes(ymin = Mean - SE, ymax = Mean + SE, fill=condition_label), alpha = 0.2, color = NA) +  # Shaded SE region
  scale_x_continuous(
    trans = "log10",  # Log-transform x-axis
    breaks = 10^seq(-7, -4, by = 1),  # Major breaks
    minor_breaks = 10^seq(-7, -4, by = 0.5),  # Finer minor grid
    labels = trans_format("log10", math_format(10^.x))  # LaTeX-style labels
  ) +
  scale_y_continuous(breaks = seq(floor(min(test$Mean - test$SE)),
                                  ceiling(max(test$Mean + test$SE)), by = 0.05))+
  annotation_logticks(sides = "b", colour="grey80") +
  theme_minimal() +
  theme(
    text = element_text(color = "black", size=8),  # Darker text
    axis.text = element_text(color = "black", size=8),  
    axis.title = element_text(color = "black", size=8),
    legend.background = element_rect(fill = alpha("white", 0.7), color = NA)) + 
  guides(fill = guide_legend(title=""),
         color = guide_legend(title=""),
         linetype = guide_legend(title=""))+
  labs(x = TeX(r"( $\textit{\alpha_{EC \rightarrow DG}}$ )"), y = "DG Similarity")+
  scale_color_viridis_d() +  # Apply Viridis to line colors
  scale_fill_viridis_d()


#Freezing ratio difference=====================================
files <- list.files(path = "demo", 
                    pattern = "^250605.*behavior\\.csv$")

df_combined <- map_dfr(files, function(file) {
  # Extract learning rate from filename
  lr_val <- as.numeric(str_match(file, "_([0-9.e-]+)lr")[, 2])
  # Extract condition type
  condition <- str_extract(file, "(mature|young)_\\dadd")
  # Match condition to label
  condition_label <- case_when(
    condition == "mature_0add" ~ "No neurogenesis",
    condition == "mature_1add" ~ "Neurogenesis - Mature",
    condition == "young_1add"  ~ "Neurogenesis - Young",
    TRUE ~ "Unknown"
  )
  # Read and return with metadata
  read_csv(paste0("demo\\", file)) %>%
    mutate(
      lr = lr_val,
      condition = condition,
      condition_label = condition_label,
      file_name = file
    )
})

df_combined$`Discrimination`=100*(df_combined$A-df_combined$B)
df_combined$X=df_combined$...1 - 3

df <- df_combined[df_combined$X>0,] %>%
  group_by(lr, condition_label) %>%
  mutate(replication = rep(1:(n() / 14), each = 14)) %>%
  ungroup()

rep_means <- df %>%
  group_by(lr, condition_label, replication) %>%
  summarise(mean_cosine = mean(Discrimination, na.rm = TRUE), .groups = "drop")

test <- rep_means %>%
  group_by(lr, condition_label) %>%
  summarise(
    Mean = mean(mean_cosine),
    SE = sd(mean_cosine) / sqrt(n()),
    .groups = "drop"
  )

ggplot(test, aes(x = lr, y = Mean, group=condition_label, color=condition_label, linetype = condition_label)) +
  geom_line() +  # Line plot
  geom_ribbon(aes(ymin = Mean - SE, ymax = Mean + SE, fill=condition_label), alpha = 0.2, color = NA) +  # Shaded SE region
  scale_x_continuous(
    trans = "log10",  # Log-transform x-axis
    breaks = 10^seq(-7, -4, by = 1),  # Major breaks
    minor_breaks = 10^seq(-7, -4, by = 0.5),  # Finer minor grid
    labels = trans_format("log10", math_format(10^.x))  # LaTeX-style labels
  ) +
  scale_y_continuous(breaks = seq(floor(min(test$Mean - test$SE)),
                                  ceiling(max(test$Mean + test$SE)), by = 2))+
  annotation_logticks(sides = "b", colour="grey80") +
  theme_minimal() +
  theme(
    text = element_text(color = "black", size=8),  # Darker text
    axis.text = element_text(color = "black", size=8),  
    axis.title = element_text(color = "black", size=8),
    legend.background = element_rect(fill = alpha("white", 0.7), color = NA)) + 
  guides(fill = guide_legend(title=""),
         color = guide_legend(title=""),
         linetype = guide_legend(title=""))+
  labs(x = TeX(r"( $\textit{\alpha_{EC \rightarrow DG}}$ )"), y = "% freezing (A - B)")+
  scale_color_viridis_d() +  # Apply Viridis to line colors
  scale_fill_viridis_d()

#Sparsity=============================
file <- list.files(path = "demo", 
                   pattern = "^250623_2.5e-05lr_50rep")
test = read.csv(paste0("demo\\",file))

df_summary <- test %>%
  group_by(X) %>%
  summarise(
    mean_Frac = mean(Fraction, na.rm = TRUE),
    sem_Frac  = sd(Fraction,   na.rm = TRUE) / sqrt(n())
  )

set.seed(123)  # for reproducibility
df_summary <- df_summary %>%
  sample_n(200)  # adjust number as needed

ggplot(df_summary, aes(x = X/10000, y = mean_Frac, group = 1)) +
  geom_line(color = "blue") +
  geom_ribbon(aes(ymin = mean_Frac - sem_Frac,
                  ymax = mean_Frac + sem_Frac),
              alpha = 0.2, color = NA) +
  labs(x = TeX(r"( $Time\,step\,(\times 10^4)$ )"),
       y = "Active fraction") +
  scale_y_continuous(
    breaks = seq(0,1,0.5),limits = c(0,1))+
  theme(
    text = element_text(size = 8, color = 'black'),
    axis.text.x = element_text(color = "black", size = 8),
    axis.text.y = element_text(color = "black", size = 8))+
  theme_minimal()

#PCA==========================
file <- list.files(path = "demo", 
                   pattern = "^250618_baseline_2.5e-05lr_50rep")
test = read.csv(paste0("demo\\",file))
test = test[-1]

PC = 10
test = test[,1:(PC+1)]

df_long <- test %>%
  group_by(Day) %>%
  mutate(replication = row_number()) %>%
  pivot_longer(cols = starts_with("PC"),
               names_to = "PC",
               values_to = "value") %>%
  ungroup() %>%
  mutate(PC = as.numeric(gsub("PC.", "", PC)))

summary_df <- df_long %>%
  group_by(Day, PC) %>%
  summarise(
    mean = mean(value),
    sem = sd(value) / sqrt(n()),
    .groups = "drop"
  )

# Plot mean ± SEM
ggplot(summary_df[summary_df$Day!=0,], aes(x = PC, y = mean, color = as.factor(Day))) +
  geom_point()+
  geom_line(alpha = 0.5) +
  geom_ribbon(aes(ymin = mean - sem, ymax = mean + sem, fill = as.factor(Day)),
              alpha = 0.2, color = NA) +
  labs(
    title = "",
    x = "PC",
    y = "Explained Variance",
    color = "Day",
    fill = "Day"
  ) +
  scale_color_viridis_d() +  
  scale_fill_viridis_d()+
  scale_x_continuous(breaks = scales::pretty_breaks())+
  theme_minimal()+
  theme(
    text = element_text(color = "black", size=8),  # Darker text
    axis.text = element_text(color = "black", size=8))

