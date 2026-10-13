# Figures for the technical report (ggplot2). Reads only result tables, so it runs in seconds.
#   pixi run -e r Rscript scripts/R/report_figures.R      (writes report/figures/fig*.png)
#
# Inputs: results/tables/{heldout_violin_per_cell, inputs_heldout_nspe_per_cell, a11_all_models_per_condition}.csv
# and data/processed/shared_model/violin_per_enhancer.csv.gz (per-enhancer correlations, written by 15e; too large
# to commit). A figure whose input is missing is skipped with a message.

suppressPackageStartupMessages(library(ggplot2))

tables <- file.path("results", "tables")
figures <- file.path("report", "figures")
dir.create(figures, showWarnings = FALSE, recursive = TRUE)
read <- function(name) read.csv(file.path(tables, name), stringsAsFactors = FALSE)
have <- function(path) {
  ok <- file.exists(path)
  if (!ok) cat("skip: missing", path, "\n")
  ok
}

INK <- "#14213D"; MUTED <- "#5B6474"; GRID <- "#E6E4DE"; PAPER <- "#FAFAF7"
RF <- "#E8683A"            # PERD-RF
GREY <- "#8A8F98"          # controls
ACT <- "#7B4FD6"           # PERD-RF + TF activity + motifs
PAIR <- "#2A78D6"          # pair model

# Arimo is a free font with Arial's metrics; it is used if fonts/arimo-{400,700}.ttf are present.
FONT <- "sans"
if (file.exists("fonts/arimo-400.ttf") && file.exists("fonts/arimo-700.ttf") && requireNamespace("systemfonts", quietly = TRUE)) {
  systemfonts::register_font("Arial", plain = "fonts/arimo-400.ttf", bold = "fonts/arimo-700.ttf")
  FONT <- "Arial"
}

theme_report <- function(base = 17) {
  theme_minimal(base_size = base, base_family = FONT) %+replace%
    theme(plot.background = element_rect(fill = PAPER, colour = NA), panel.background = element_rect(fill = PAPER, colour = NA),
          panel.grid.major = element_line(colour = GRID, linewidth = 0.45), panel.grid.minor = element_blank(),
          axis.text = element_text(colour = MUTED, size = 15), axis.title = element_text(colour = INK),
          axis.line.x = element_line(colour = MUTED, linewidth = 0.5), axis.ticks.x = element_line(colour = MUTED, linewidth = 0.4),
          legend.position = "top", legend.justification = "left", legend.title = element_blank(),
          legend.text = element_text(colour = INK, size = 15),
          strip.text = element_text(colour = INK, face = "bold", size = 17, hjust = 0, margin = margin(b = 8)),
          plot.margin = margin(16, 24, 12, 12))
}

save <- function(plot, name, width, height) {
  path <- file.path(figures, name)
  if (requireNamespace("ragg", quietly = TRUE)) {
    ggsave(path, plot, width = width, height = height, dpi = 300, device = ragg::agg_png, bg = PAPER)
  } else {
    ggsave(path, plot, width = width, height = height, dpi = 300, bg = PAPER)
  }
  cat("wrote", path, "\n")
}

nspe_x <- scale_x_log10(breaks = c(0.25, 0.5, 1, 2, 4), labels = c("0.25", "0.5", "1", "2", "4"))

DS5 <- c(encode_dex = "A549\ndexamethasone", encode_k562 = "K562\n10 drugs", encode_tcell = "T-cell\nactivation",
         encode_hct116 = "HCT116\ndegrons", encode_sknsh = "SK-N-SH\nretinoic acid*")
PRED <- c(frozen_rf = "PERD-RF (RNA only)", tfadd_rf = "+ TF activity", tfmotif_rf = "+ motifs", tfboth_rf = "+ TF activity + motifs",
          E4 = "Pair model", B0 = "Starting openness only")

# ---- Figure 1: accuracy on unseen tissues (57 test cells; random split vs tissue held out vs shared RF) ----
vfile <- file.path("data", "processed", "shared_model", "violin_per_enhancer.csv.gz")
if (have(vfile) && have(file.path(tables, "heldout_violin_per_cell.csv"))) {
  grp_lab <- function(model, scheme) ifelse(model == "PERD-RF" & scheme == "Random 110/57 split", "PERD-RF\nrandom split",
                                     ifelse(model == "PERD-RF", "PERD-RF\ntissue held out", "Shared RF\ntissue held out"))
  grp_col <- c("PERD-RF\nrandom split" = "#F4B393", "PERD-RF\ntissue held out" = RF, "Shared RF\ntissue held out" = GREY)
  e <- subset(read.csv(gzfile(vfile), stringsAsFactors = FALSE), cells == "57 test cells" & !is.na(cross_cell_r))
  cl <- subset(read("heldout_violin_per_cell.csv"), cells == "57 test cells")
  long <- rbind(
    data.frame(grp = grp_lab(e$model, e$scheme), metric = "Correlation per enhancer\n(across cells)", value = e$cross_cell_r),
    data.frame(grp = grp_lab(cl$model, cl$scheme), metric = "Correlation per cell\n(across enhancers)", value = cl$cross_enhancer_r),
    data.frame(grp = grp_lab(cl$model, cl$scheme), metric = "NSPE per cell\n(lower = better)", value = cl$nspe))
  long$grp <- factor(long$grp, levels = names(grp_col))
  long$metric <- factor(long$metric, levels = unique(long$metric))
  med <- aggregate(value ~ grp + metric, data = long, FUN = median)
  rng <- do.call(rbind, lapply(split(long$value, long$metric), function(v) data.frame(lo = min(v), hi = max(v))))
  rng$metric <- factor(rownames(rng), levels = levels(long$metric))
  med$y <- (rng$hi + (rng$hi - rng$lo) * 0.16)[match(med$metric, rng$metric)]
  p <- ggplot(long, aes(grp, value, fill = grp)) +
    geom_violin(colour = NA, scale = "width", width = 0.68, trim = FALSE, adjust = 0.8, alpha = 0.9) +
    geom_boxplot(width = 0.1, outlier.shape = NA, fill = "white", colour = INK, linewidth = 0.5) +
    geom_text(data = med, aes(y = y, label = sprintf("%.2f", value)), colour = INK, size = 5.4, fontface = "bold", family = FONT) +
    facet_wrap(~metric, nrow = 1, scales = "free_y") +
    scale_fill_manual(values = grp_col, guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(0.04, 0.08))) +
    labs(x = NULL, y = NULL) +
    theme_report() + theme(panel.grid.major.x = element_blank(), panel.spacing = unit(4.5, "lines"),
                           axis.text.x = element_text(colour = INK, size = 14, lineheight = 0.95))
  save(p, "fig1_heldout_tissues.png", 18, 7.6)
}

a11_path <- file.path(tables, "a11_all_models_per_condition.csv")
a11 <- NULL
if (have(a11_path)) {
  a11 <- read("a11_all_models_per_condition.csv")
  a11$ds <- factor(DS5[a11$dataset], levels = rev(DS5))
}

# ---- Figure 2: frozen PERD-RF on the 40 perturbation pairs, NSPE before / after / change ----
if (!is.null(a11)) {
  f <- subset(a11, predictor == "frozen_rf")
  side_lab <- c(nspe_before = "Before (control)", nspe_after = "After (treated)", nspe_change = "Change (after − before)")
  long <- do.call(rbind, lapply(names(side_lab), function(s) data.frame(ds = f$ds, side = side_lab[[s]], nspe = f[[s]])))
  long$side <- factor(long$side, levels = side_lab)
  m <- aggregate(nspe ~ ds + side, data = long, FUN = median)
  p <- ggplot(long, aes(nspe, ds)) +
    geom_vline(xintercept = 1, linetype = "dashed", colour = MUTED, linewidth = 0.6) +
    geom_point(colour = RF, alpha = 0.3, size = 2.4, position = position_jitter(height = 0.1, width = 0, seed = 1)) +
    geom_point(data = m, colour = RF, fill = PAPER, shape = 21, size = 5.6, stroke = 2.2) +
    geom_text(data = m, aes(label = sprintf("%.2f", nspe)), vjust = -1.25, colour = INK, size = 5, fontface = "bold", family = FONT) +
    facet_wrap(~side, nrow = 1) + nspe_x +
    labs(x = "NSPE", y = NULL) +
    theme_report() + theme(panel.spacing = unit(2.5, "lines"), panel.grid.major.y = element_blank(), axis.text.y = element_text(colour = INK))
  save(p, "fig2_before_after_change.png", 15, 7)
}

# ---- Figure 3: unseen tissues, NSPE per held-out cell for each input set ----
if (have(file.path(tables, "inputs_heldout_nspe_per_cell.csv"))) {
  v <- read("inputs_heldout_nspe_per_cell.csv")
  lab <- c("Frozen PERD-RF" = "PERD-RF\n(RNA only)", "+ shuffled-target activity" = "+ shuffled\nactivity (control)",
           "+ TF activity" = "+ TF activity", "+ motif activity" = "+ motifs", "+ both" = "+ TF activity\n+ motifs")
  col <- c(RF, GREY, "#B9A3EE", "#9A7BE6", ACT); names(col) <- lab
  v$grp <- factor(lab[v$variant], levels = lab)
  m <- aggregate(nspe ~ grp, data = v, FUN = median)
  p <- ggplot(v, aes(grp, nspe, fill = grp)) +
    geom_violin(colour = NA, scale = "width", width = 0.7, trim = FALSE, adjust = 0.8, alpha = 0.9) +
    geom_boxplot(width = 0.1, outlier.shape = NA, fill = "white", colour = INK, linewidth = 0.5) +
    geom_text(data = m, aes(y = max(v$nspe) * 1.08, label = sprintf("%.3f", nspe)), colour = INK, size = 5.2, fontface = "bold", family = FONT) +
    scale_fill_manual(values = col, guide = "none") +
    labs(x = NULL, y = "NSPE") +
    theme_report() + theme(panel.grid.major.x = element_blank(), axis.text.x = element_text(colour = INK))
  save(p, "fig3_inputs_heldout.png", 12, 6.4)
}

# ---- Figure 4: change NSPE on the 40 pairs, PERD-RF vs + TF activity + motifs ----
if (!is.null(a11)) {
  v <- subset(a11, predictor %in% c("frozen_rf", "tfboth_rf"))
  lv <- unname(PRED[c("frozen_rf", "tfboth_rf")])
  v$pr <- factor(PRED[v$predictor], levels = rev(lv))
  cols <- c(RF, ACT); names(cols) <- lv
  m <- aggregate(nspe_change ~ ds + pr, data = v, FUN = median)
  pd <- position_dodge(width = 0.7)
  p <- ggplot(v, aes(ds, nspe_change, fill = pr, colour = pr)) +
    geom_hline(yintercept = 1, linetype = "dashed", colour = MUTED, linewidth = 0.6) +
    geom_boxplot(position = pd, width = 0.6, alpha = 0.3, outlier.size = 1.4, outlier.alpha = 0.7, linewidth = 0.6) +
    geom_text(data = m, aes(y = 3.0, label = sprintf("%.2f", nspe_change), group = pr), position = pd, hjust = 1, size = 4.6,
              fontface = "bold", family = FONT, colour = INK, show.legend = FALSE) +
    annotate("text", x = length(levels(v$ds)) + 0.55, y = 3.0, label = "median", hjust = 1, size = 4.4, colour = MUTED, family = FONT) +
    scale_fill_manual(values = cols, breaks = lv) + scale_colour_manual(values = cols, breaks = lv) +
    scale_y_log10(breaks = c(1, 1.25, 1.5, 2, 2.5), labels = c("1", "1.25", "1.5", "2", "2.5")) +
    coord_flip(clip = "off") +
    labs(x = NULL, y = "NSPE of the change") +
    theme_report() + theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(colour = INK)) +
    guides(fill = guide_legend(nrow = 1), colour = guide_legend(nrow = 1))
  save(p, "fig4_inputs_change.png", 14, 7.2)
}

# ---- Figure 5: the pair model, as a flow diagram ----
local({
  steps <- data.frame(
    x = c(2.1, 6.9, 11.7, 16.5),
    head = c("Real before/after pairs", "Training table", "One shared\nrandom forest", "Prediction"),
    body = c("16 perturbations\n(40 pairs)", "one row per enhancer per pair:\ndrug-induced changes\n+ starting openness",
             "learns\ninputs → change in openness", "change in openness\nfor a perturbation\nit never saw"),
    fill = c("#EEF1F5", "#F1ECFB", "#E4F1E8", "#FCE3D6"), line = c("#C9D0DA", "#D6C8F5", "#9CCBAA", "#F0B89C"))
  p <- ggplot() + coord_cartesian(xlim = c(0, 18.6), ylim = c(0, 6), expand = FALSE)
  for (i in seq_len(nrow(steps))) {
    p <- p + annotate("rect", xmin = steps$x[i] - 1.95, xmax = steps$x[i] + 1.95, ymin = 2.2, ymax = 5.2, fill = steps$fill[i], colour = steps$line[i], linewidth = 0.9) +
      annotate("text", x = steps$x[i], y = 4.5, label = steps$head[i], size = 6.2, fontface = "bold", colour = INK, family = FONT, lineheight = 0.9) +
      annotate("text", x = steps$x[i], y = 3.35, label = steps$body[i], size = 4.9, colour = INK, family = FONT, lineheight = 0.95)
    if (i < nrow(steps)) p <- p + annotate("segment", x = steps$x[i] + 2.0, xend = steps$x[i + 1] - 2.0, y = 3.7, yend = 3.7, colour = GREY,
                                           linewidth = 1.4, arrow = arrow(length = unit(0.32, "cm"), type = "closed"))
  }
  p <- p + annotate("curve", x = 16.5, y = 2.1, xend = 6.9, yend = 2.1, curvature = -0.22, colour = ACT, linewidth = 1, linetype = "dashed",
                    arrow = arrow(length = unit(0.28, "cm"), type = "closed")) +
    annotate("text", x = 11.7, y = 0.95, label = "leave one perturbation out: train on 15, predict the 16th, repeat 16×", size = 5,
             fontface = "bold", colour = ACT, family = FONT) +
    annotate("text", x = 9.3, y = 5.75, label = "Control model: same forest, but only starting openness + fixed enhancer properties (no drug information)",
             size = 4.8, colour = GREY, family = FONT) +
    theme_void() + theme(plot.background = element_rect(fill = PAPER, colour = NA), plot.margin = margin(8, 8, 8, 8))
  save(p, "fig5_pair_model.png", 16, 5.4)
})

# ---- Figure 6: median change NSPE, every model x every dataset ----
if (!is.null(a11)) {
  g <- subset(a11, predictor %in% names(PRED))
  unit <- aggregate(nspe_change ~ dataset + unit + predictor, data = g, FUN = median)
  by_ds <- aggregate(nspe_change ~ dataset + predictor, data = g, FUN = median)
  all16 <- aggregate(nspe_change ~ predictor, data = unit, FUN = median); all16$dataset <- "all"
  grid <- rbind(by_ds, all16[c("dataset", "predictor", "nspe_change")])
  rows <- c(DS5, all = "All 16\nperturbations")
  grid$row <- factor(rows[grid$dataset], levels = rev(rows))
  grid$col <- factor(PRED[grid$predictor], levels = PRED)
  p <- ggplot(grid, aes(col, row, fill = log2(nspe_change))) +
    geom_tile(colour = PAPER, linewidth = 2) +
    geom_text(aes(label = sprintf("%.2f", nspe_change)), colour = INK, size = 6, fontface = "bold", family = FONT) +
    scale_fill_gradient2(low = PAIR, mid = "#F4F1EA", high = RF, midpoint = 0, limits = c(-0.6, 0.6), oob = scales::squish,
                         breaks = log2(c(0.75, 1, 1.4)), labels = c("0.75", "1", "1.4"), name = "NSPE of the change (blue = beats 'no change')") +
    scale_x_discrete(position = "top", labels = function(x) gsub(" \\+ ", "\n+ ", gsub("^Starting openness only$", "Starting openness\nonly (control)", x))) +
    labs(x = NULL, y = NULL) +
    theme_report() + theme(panel.grid = element_blank(), axis.text.x = element_text(colour = INK, size = 14, face = "bold"),
                           axis.text.y = element_text(colour = INK), legend.position = "bottom", legend.key.width = unit(3, "lines"),
                           legend.title = element_text(colour = INK, size = 14), axis.line = element_blank(), axis.ticks = element_blank())
  save(p, "fig6_all_models_grid.png", 15, 8)
}
