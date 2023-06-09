# load the necessary packages
library(readr)
library(dplyr)
library(ggplot2)
library(forcats)
library(viridis)

# define the paths to csv files containing results on your local computer
load_data_path <- '/Users/neo/Documents/Research/CP/dev_wsi_patch_ssl/benchmarking/benp_results/load_speed_test_benp_results.csv'
save_data_path <- '/Users/neo/Documents/Research/CP/dev_wsi_patch_ssl/benchmarking/benp_results/save_speed_test_benp_results.csv'


load_df <- read.csv(load_data_path)
save_df <- read.csv(save_data_path)

load_df <- load_df %>% 
  rename(eff_n_patches = eff_n_patches...cumulative,
         h5 = h5...cumulative,
         folder_img = folder.images...cumulative,
         folder_np = folder.saver.numpy_arr...cumulative,
         folder_np_par = folder.saver.numpy_arr.parallel...cumulative,
         folder_np_compressed = folder.saver.numpy_arr_compressed...cumulative,
         folder_np_compressed_par = folder.saver.numpy_arr_compressed.parallel...cumulative,
         folder_csv = folder.saver.csv...cumulative,
         folder_csv_par = folder.saver.csv.parallel...cumulative,
         folder_parquet = folder.saver.parquet...cumulative,
         folder_parquet_par = folder.saver.parquet.parallel...cumulative,
         folder_pil_png = folder.saver.pil_image_png...cumulative,
         folder_pil_png_par = folder.saver.pil_image_png.parallel...cumulative,
         folder_pil_jpg = folder.saver.pil_image_jpg...cumulative,
         folder_pil_jpg_par = folder.saver.pil_image_jpg.parallel...cumulative,
         folder_torch = folder.saver.torch...cumulative,
         folder_torch_par = folder.saver.torch.parallel...cumulative,
         openslide = openslide...cumulative,
         openslide_par_ck1 = openslide.parallel..chunksize.1...cumulative,
         folder_par_ck1 = folder.parallel..chunksize.1...cumulative,
         openslide_par_ck10 = openslide.parallel..chunksize.10...cumulative,
         folder_par_ck10 = folder.parallel..chunksize.10...cumulative,
         openslide_par_ck100 = openslide.parallel..chunksize.100...cumulative,
         folder_par_ck1000 = folder.parallel..chunksize.100...cumulative
         ) %>%
  select(eff_n_patches,
         h5, 
         folder_img,
         folder_np,
         folder_np_par,
         folder_np_compressed,
         folder_np_compressed_par,
         folder_csv,
         folder_csv_par,
         folder_parquet,
         folder_parquet_par,
         folder_pil_png,
         folder_pil_png_par,
         folder_pil_jpg,
         folder_pil_jpg_par,
         folder_torch,
         folder_torch_par,
         openslide,
         openslide_par_ck1,
         folder_par_ck1,
         openslide_par_ck10,
         folder_par_ck10,
         openslide_par_ck100,
         folder_par_ck1000)

save_df <- save_df %>%
  rename(eff_n_patches = eff_n_patches...cumulative,
         folder_image_patch_v1 = folder.image.patches.v1...cumulative,
         folder_image_patch_v1_par = folder.image.patches.v1..parallel...cumulative,
         folder_np = folder.saver.numpy_arr...cumulative,
         folder_np_par = folder.saver.numpy_arr.parallel...cumulative,
         folder_np_compressed = folder.saver.numpy_arr_compressed...cumulative,
         folder_np_compressed_par = folder.saver.numpy_arr_compressed.parallel...cumulative,
         folder_csv = folder.saver.csv...cumulative,
         folder_csv_par = folder.saver.csv.parallel...cumulative,
         folder_parquet = folder.saver.parquet...cumulative,
         folder_parquet_par = folder.saver.parquet.parallel...cumulative,
         folder_pil_png = folder.saver.pil_image_png...cumulative,
         folder_pil_png_par = folder.saver.pil_image_png.parallel...cumulative,
         folder_pil_jpg = folder.saver.pil_image_jpg...cumulative,
         folder_pil_jpg_par = folder.saver.pil_image_jpg.parallel...cumulative,
         folder_torch = folder.saver.torch...cumulative,
         folder_torch_par = folder.saver.torch.parallel...cumulative,
         wsi_file_size = wsi_svs...file_size,
         h5_file_size = h5...file_size,
         np_file_size = numpy_arr...file_size,
         np_compressed_file_size = numpy_arr_compressed...file_size,
         csv_file_size = csv...file_size,
         parquet_file_size = parquet...file_size,
         pil_png_file_size = pil_image_png...file_size,
         pil_jpg_file_size = pil_image_jpg...file_size,
         torch_file_size = torch...file_size
         )

size_df <- save_df %>%
  select(wsi_file_size,
         h5_file_size,
         np_file_size,
         np_compressed_file_size,
         csv_file_size,
         parquet_file_size,
         pil_png_file_size,
         pil_jpg_file_size,
         torch_file_size
         )

save_df <- save_df %>%
  select(eff_n_patches,
         folder_image_patch_v1,
         folder_image_patch_v1_par,
         folder_np,
         folder_np_par,
         folder_np_compressed,
         folder_np_compressed_par,
         folder_csv,
         folder_csv_par,
         folder_parquet,
         folder_parquet_par,
         folder_pil_png,
         folder_pil_png_par,
         folder_pil_jpg,
         folder_pil_jpg_par,
         folder_torch,
         folder_torch_par)









################################################################################
# PLOTTING THE SAVE TIME
################################################################################

# select the required columns and calculate their averages
df_average <- save_df %>%
  select(
    # folder_image_patch_v1,
    # folder_image_patch_v1_par,
    folder_np,
    folder_np_par,
    folder_np_compressed,
    folder_np_compressed_par,
    folder_csv,
    folder_csv_par,
    folder_parquet,
    folder_parquet_par,
    folder_pil_png,
    folder_pil_png_par,
    folder_pil_jpg,
    folder_pil_jpg_par,
    folder_torch,
    folder_torch_par
  ) %>%
  summarise(across(everything(), mean, na.rm = TRUE))

# transform the data to long format
df_average_long <- tidyr::pivot_longer(df_average, everything(), names_to = "Column", values_to = "Average")

# reorder and reverse the factor levels of 'Column' according to 'Average'
df_average_long$Column <- df_average_long$Column %>%
  fct_reorder(-df_average_long$Average)

# define the colors
bar_color <- "#acb1b6"  # light grey with a slight blue undertone
background_color <- "#1a1f27"  # darker grey with a slight blue undertone

# create the bar plot
save_time_plot <- ggplot(df_average_long, aes(y = Column, x = Average)) +
  geom_col(fill = bar_color, width = 0.5) +
  scale_x_log10(labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  xlab("Average saving time (log10 scale, original unit in seconds)") +
  ylab("File formats (all)") +
  theme_minimal(base_size = 12, base_family = "") +
  theme(plot.background = element_rect(fill = background_color),
        panel.background = element_rect(fill = background_color),
        panel.grid.major = element_line(color = "#333333"),
        panel.grid.minor = element_line(color = "#333333"),
        text = element_text(color = bar_color),
        axis.text = element_text(color = bar_color),
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
        axis.title = element_text(face = "bold", color = bar_color),
        plot.margin = margin(1, 1, 1, 3, "cm"))  # Adjust the margin











# select the required columns and calculate their averages
df_average <- save_df %>%
  select(
    # folder_image_patch_v1,
    # folder_image_patch_v1_par,
    folder_np_par,
    folder_np_compressed_par,
    folder_csv_par,
    folder_parquet_par,
    folder_pil_png_par,
    folder_pil_jpg_par,
    folder_torch_par
  ) %>%
  summarise(across(everything(), mean, na.rm = TRUE))

# transform the data to long format
df_average_long <- tidyr::pivot_longer(df_average, everything(), names_to = "Column", values_to = "Average")

# reorder and reverse the factor levels of 'Column' according to 'Average'
df_average_long$Column <- df_average_long$Column %>%
  fct_reorder(-df_average_long$Average)

# define the colors
bar_color <- "#acb1b6"  # light grey with a slight blue undertone
background_color <- "#1a1f27"  # darker grey with a slight blue undertone

# create the bar plot
save_time_plot_par <- ggplot(df_average_long, aes(y = Column, x = Average)) +
  geom_col(fill = bar_color, width = 0.5) +
  scale_x_log10(labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  xlab("Average saving time (log10 scale, original unit in seconds)") +
  ylab("File formats (parallel only)") +
  theme_minimal(base_size = 12, base_family = "") +
  theme(plot.background = element_rect(fill = background_color),
        panel.background = element_rect(fill = background_color),
        panel.grid.major = element_line(color = "#333333"),
        panel.grid.minor = element_line(color = "#333333"),
        text = element_text(color = bar_color),
        axis.text = element_text(color = bar_color),
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
        axis.title = element_text(face = "bold", color = bar_color),
        plot.margin = margin(1, 1, 1, 3, "cm"))  # Adjust the margin











# select the required columns and calculate their averages
df_average <- save_df %>%
  select(
    # folder_image_patch_v1,
    # folder_image_patch_v1_par,
    folder_np,
    folder_np_compressed,
    folder_csv,
    folder_parquet,
    folder_pil_png,
    folder_pil_jpg,
    folder_torch,
  ) %>%
  summarise(across(everything(), mean, na.rm = TRUE))

# transform the data to long format
df_average_long <- tidyr::pivot_longer(df_average, everything(), names_to = "Column", values_to = "Average")

# reorder and reverse the factor levels of 'Column' according to 'Average'
df_average_long$Column <- df_average_long$Column %>%
  fct_reorder(-df_average_long$Average)

# define the colors
bar_color <- "#acb1b6"  # light grey with a slight blue undertone
background_color <- "#1a1f27"  # darker grey with a slight blue undertone

# create the bar plot
save_time_plot_nonpar <- ggplot(df_average_long, aes(y = Column, x = Average)) +
  geom_col(fill = bar_color, width = 0.5) +
  scale_x_log10(labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  xlab("Average saving time (log10 scale, original unit in seconds)") +
  ylab("File formats (non-parallel only)") +
  theme_minimal(base_size = 12, base_family = "") +
  theme(plot.background = element_rect(fill = background_color),
        panel.background = element_rect(fill = background_color),
        panel.grid.major = element_line(color = "#333333"),
        panel.grid.minor = element_line(color = "#333333"),
        text = element_text(color = bar_color),
        axis.text = element_text(color = bar_color),
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
        axis.title = element_text(face = "bold", color = bar_color),
        plot.margin = margin(1, 1, 1, 3, "cm"))  # Adjust the margin




################################################################################
# PLOTTING THE LOAD TIME
################################################################################

# select the required columns and calculate their averages
df_average <- load_df %>%
  select(
    h5, 
    folder_img,
    folder_np,
    folder_np_par,
    folder_np_compressed,
    folder_np_compressed_par,
    folder_csv,
    folder_csv_par,
    folder_parquet,
    folder_parquet_par,
    folder_pil_png,
    folder_pil_png_par,
    folder_pil_jpg,
    folder_pil_jpg_par,
    folder_torch,
    folder_torch_par,
    openslide,
    openslide_par_ck1,
    folder_par_ck1,
    openslide_par_ck10,
    folder_par_ck10,
    openslide_par_ck100,
    folder_par_ck1000
  ) %>%
  summarise(across(everything(), mean, na.rm = TRUE))

# transform the data to long format
df_average_long <- tidyr::pivot_longer(df_average, everything(), names_to = "Column", values_to = "Average")

# reorder and reverse the factor levels of 'Column' according to 'Average'
df_average_long$Column <- df_average_long$Column %>%
  fct_reorder(-df_average_long$Average)

# define the colors
bar_color <- "#acb1b6"  # light grey with a slight blue undertone
background_color <- "#1a1f27"  # darker grey with a slight blue undertone

# create the bar plot
load_time_plot <- ggplot(df_average_long, aes(y = Column, x = Average)) +
  geom_col(fill = bar_color, width = 0.5) +
  scale_x_log10(labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  xlab("Average loading time (log10 scale, original unit in seconds)") +
  ylab("File format") +
  theme_minimal(base_size = 12, base_family = "") +
  theme(plot.background = element_rect(fill = background_color),
        panel.background = element_rect(fill = background_color),
        panel.grid.major = element_line(color = "#333333"),
        panel.grid.minor = element_line(color = "#333333"),
        text = element_text(color = bar_color),
        axis.text = element_text(color = bar_color),
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
        axis.title = element_text(face = "bold", color = bar_color),
        plot.margin = margin(1, 1, 1, 3, "cm"))  # Adjust the margin








# select the required columns and calculate their averages
df_average <- load_df %>%
  select(
    folder_np_par,
    folder_np_compressed_par,
    folder_csv_par,
    folder_parquet_par,
    folder_pil_png_par,
    folder_pil_jpg_par,
    folder_torch_par,
    openslide_par_ck1,
    folder_par_ck1,
    openslide_par_ck10,
    folder_par_ck10,
    openslide_par_ck100,
    folder_par_ck1000
  ) %>%
  summarise(across(everything(), mean, na.rm = TRUE))

# transform the data to long format
df_average_long <- tidyr::pivot_longer(df_average, everything(), names_to = "Column", values_to = "Average")

# reorder and reverse the factor levels of 'Column' according to 'Average'
df_average_long$Column <- df_average_long$Column %>%
  fct_reorder(-df_average_long$Average)

# define the colors
bar_color <- "#acb1b6"  # light grey with a slight blue undertone
background_color <- "#1a1f27"  # darker grey with a slight blue undertone

# create the bar plot
load_time_plot_par <- ggplot(df_average_long, aes(y = Column, x = Average)) +
  geom_col(fill = bar_color, width = 0.5) +
  scale_x_log10(labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  xlab("Average loading time (log10 scale, original unit in seconds)") +
  ylab("File format (parallel only)") +
  theme_minimal(base_size = 12, base_family = "") +
  theme(plot.background = element_rect(fill = background_color),
        panel.background = element_rect(fill = background_color),
        panel.grid.major = element_line(color = "#333333"),
        panel.grid.minor = element_line(color = "#333333"),
        text = element_text(color = bar_color),
        axis.text = element_text(color = bar_color),
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
        axis.title = element_text(face = "bold", color = bar_color),
        plot.margin = margin(1, 1, 1, 3, "cm"))  # Adjust the margin





# select the required columns and calculate their averages
df_average <- load_df %>%
  select(
    h5, 
    folder_img,
    folder_np,
    folder_np_compressed,
    folder_csv,
    folder_parquet,
    folder_pil_png,
    folder_pil_jpg,
    folder_torch,
    openslide,
  ) %>%
  summarise(across(everything(), mean, na.rm = TRUE))

# transform the data to long format
df_average_long <- tidyr::pivot_longer(df_average, everything(), names_to = "Column", values_to = "Average")

# reorder and reverse the factor levels of 'Column' according to 'Average'
df_average_long$Column <- df_average_long$Column %>%
  fct_reorder(-df_average_long$Average)

# define the colors
bar_color <- "#acb1b6"  # light grey with a slight blue undertone
background_color <- "#1a1f27"  # darker grey with a slight blue undertone

# create the bar plot
load_time_plot_nonpar <- ggplot(df_average_long, aes(y = Column, x = Average)) +
  geom_col(fill = bar_color, width = 0.5) +
  scale_x_log10(labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  xlab("Average loading time (log10 scale, original unit in seconds)") +
  ylab("File format (non-parallel only)") +
  theme_minimal(base_size = 12, base_family = "") +
  theme(plot.background = element_rect(fill = background_color),
        panel.background = element_rect(fill = background_color),
        panel.grid.major = element_line(color = "#333333"),
        panel.grid.minor = element_line(color = "#333333"),
        text = element_text(color = bar_color),
        axis.text = element_text(color = bar_color),
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
        axis.title = element_text(face = "bold", color = bar_color),
        plot.margin = margin(1, 1, 1, 3, "cm"))  # Adjust the margin



################################################################################
# PLOTTING THE FILE SIZES
################################################################################

# select the required columns and calculate their averages
df_average <- size_df %>%
  select(
    wsi_file_size,
    h5_file_size,
    np_file_size,
    np_compressed_file_size,
    csv_file_size,
    parquet_file_size,
    pil_png_file_size,
    pil_jpg_file_size,
    torch_file_size
  ) %>%
  summarise(across(everything(), mean, na.rm = TRUE))

# transform the data to long format
df_average_long <- tidyr::pivot_longer(df_average, everything(), names_to = "Column", values_to = "Average")

# reorder the factor levels of 'Column' according to 'Average'
df_average_long$Column <- df_average_long$Column %>%
  fct_reorder(-df_average_long$Average)

# define the colors
bar_color <- "#acb1b6"  # light grey with a slight blue undertone
background_color <- "#1a1f27"  # darker grey with a slight blue undertone

# create the bar plot
file_size_plot <- ggplot(df_average_long, aes(y = Column, x = Average)) +
  geom_col(fill = bar_color, width = 0.5) +
  labs(x = "Average file size (GB)",
       y = "File format") +
  theme_minimal(base_size = 12, base_family = "") +
  theme(plot.background = element_rect(fill = background_color),
        panel.background = element_rect(fill = background_color),
        panel.grid.major = element_line(color = "#333333"),
        panel.grid.minor = element_line(color = "#333333"),
        text = element_text(color = bar_color),
        axis.text = element_text(color = bar_color),
        axis.title = element_text(face = "bold", color = bar_color),
        plot.margin = margin(1, 1, 1, 3, "cm"))  # Adjust the margin





################################################################################
# DISPLAYING THE PLOT
################################################################################

save_time_plot
save_time_plot_nonpar
save_time_plot_par
load_time_plot
load_time_plot_nonpar
load_time_plot_par
file_size_plot
