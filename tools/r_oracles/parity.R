# Reference values for the probstat-versus-R parity suite.
#
# Run from the repository root:
#
#     Rscript tools/r_oracles/parity.R
#
# The script needs base R with its recommended packages (stats, MASS, nlme,
# survival) plus fitdistrplus, ismev, and e1071. Every dataset is written out
# literally below and checked against the copy shipped with R, so the
# reference values do not depend on the installed datasets. The output,
# tests/data/r_parity.json, holds the R version, the package versions, the
# datasets, and one record per case: an id, the probstat target, the exact R
# call (the string that is evaluated), the R values, the comparison tolerance,
# and the reason for that tolerance. tests/test_r_parity.py reads the file and
# repeats every case in probstat; it never calls R.
#
# Tolerance policy. A case passes when |probstat - R| <= absolute +
# relative * |R| for every value. Both sides evaluating the same closed form
# or the same special function use relative 1e-10 (often tighter). Looser
# tolerances appear only where one side stops an iteration (an optimizer, a
# root finder, or a quadrature) at a documented accuracy, and the reason
# names that algorithm. Where R and probstat use different, equally valid
# definitions, the R call is changed to the definition probstat implements
# (and the case says so in its note) or the comparison is listed under
# "skipped" with the reason.

suppressPackageStartupMessages({
  library(stats)
  library(MASS)
  library(nlme)
  library(survival)
  library(fitdistrplus)
  library(ismev)
  library(e1071)
})

args <- commandArgs(trailingOnly = TRUE)
output_path <- if (length(args) >= 1) args[[1]] else file.path("tests", "data", "r_parity.json")

# ---------------------------------------------------------------- JSON writer

json_number <- function(v) {
  if (!is.finite(v)) stop("non-finite value in a parity record")
  if (v == 0) return("0")
  for (digits in 15:17) {
    text <- formatC(v, digits = digits, format = "g")
    if (as.numeric(text) == v) break
  }
  sub("^\\s+", "", text)
}

json_string <- function(s) {
  s <- gsub("\\\\", "\\\\\\\\", s)
  s <- gsub("\"", "\\\\\"", s)
  s <- gsub("\n", "\\\\n", s)
  s <- gsub("\t", "\\\\t", s)
  paste0("\"", s, "\"")
}

json_atom <- function(v) {
  if (is.character(v)) json_string(v)
  else if (is.logical(v)) if (v) "true" else "false"
  else json_number(as.numeric(v))
}

# Named lists become objects, unnamed lists arrays; atomic vectors of length
# one become scalars unless wrapped in I(), which always gives an array.
to_json <- function(x, indent = 0) {
  pad <- strrep("  ", indent + 1)
  end <- strrep("  ", indent)
  if (is.list(x)) {
    if (length(x) == 0) return(if (is.null(names(x))) "[]" else "{}")
    items <- vapply(x, to_json, character(1), indent = indent + 1)
    if (!is.null(names(x))) {
      items <- paste0(json_string(names(x)), ": ", items)
      return(paste0("{\n", pad, paste(items, collapse = paste0(",\n", pad)), "\n", end, "}"))
    }
    return(paste0("[\n", pad, paste(items, collapse = paste0(",\n", pad)), "\n", end, "]"))
  }
  if (is.factor(x)) x <- as.character(x)
  if (length(x) == 1 && !inherits(x, "AsIs")) return(json_atom(x))
  paste0("[", paste(vapply(unclass(x), json_atom, character(1)), collapse = ", "), "]")
}

# ------------------------------------------------------------- case registry

cases <- list()
skipped <- list()

# Evaluate `call` (a string), which must return a named list of numeric
# vectors, and record it with its tolerance and probstat target.
add_case <- function(id, area, target, call, relative, absolute, reason,
                     inputs = character(), note = NULL, probstat = NULL) {
  if (id %in% vapply(cases, `[[`, character(1), "id")) stop("duplicate case id ", id)
  value <- eval(parse(text = call), envir = globalenv())
  if (!is.list(value) || is.null(names(value))) stop(id, ": the call must return a named list")
  value <- lapply(value, function(v) I(unname(as.numeric(v))))
  record <- list(
    id = id, area = area, target = target, r_call = call,
    inputs = I(inputs), r_value = value,
    tolerance = list(relative = relative, absolute = absolute),
    tolerance_reason = reason
  )
  if (!is.null(note)) record$note <- note
  if (!is.null(probstat)) record$probstat <- probstat
  cases[[length(cases) + 1]] <<- record
  invisible(value)
}

add_skip <- function(r_function, reason) {
  skipped[[length(skipped) + 1]] <<- list(r_function = r_function, reason = reason)
}

num_list <- function(x) paste(vapply(x, json_number, character(1)), collapse = ", ")

# A distribution-function case: R's d/p/q function at `points`, compared with
# the probstat law's `method` at the same points.
add_law <- function(id, law, parameters, method, r_function, r_parameters,
                    points, relative = 1e-10, absolute = 0, reason = NULL,
                    tail = "") {
  call <- sprintf("list(values = %s(c(%s), %s%s))", r_function, num_list(points),
                  r_parameters, tail)
  if (is.null(reason)) {
    reason <- "Both sides evaluate the same special function to near machine precision."
  }
  add_case(id, "distributions", paste0("probstat.distributions.", law, ".", method),
           call, relative, absolute, reason,
           probstat = list(law = law, parameters = parameters, method = method,
                           points = I(points)))
}

same_closed_form <- "Both sides evaluate the same closed form; only floating-point rounding differs."
same_special <- "Both sides evaluate the same closed form through the regularized incomplete beta or gamma function, accurate to near machine precision on each side."

# ---------------------------------------------------------------- datasets
# Literal copies of the columns of R datasets used by the cases.

sleep_extra <- c(0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0, 2, 1.9, 0.8,
1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4)
sleep_group <- c(1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L, 2L,
2L, 2L, 2L, 2L, 2L)
sleep_id <- c(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 1L, 2L, 3L, 4L, 5L,
6L, 7L, 8L, 9L, 10L)
tg_len <- c(4.2, 11.5, 7.3, 5.8, 6.4, 10, 11.2, 11.2, 5.2, 7, 16.5, 16.5,
15.2, 17.3, 22.5, 17.3, 13.6, 14.5, 18.8, 15.5, 23.6, 18.5, 33.9,
25.5, 26.4, 32.5, 26.7, 21.5, 23.3, 29.5, 15.2, 21.5, 17.6, 9.7,
14.5, 10, 8.2, 9.4, 16.5, 9.7, 19.7, 23.3, 23.6, 26.4, 20, 25.2,
25.8, 21.2, 14.5, 27.3, 25.5, 26.4, 22.4, 24.5, 24.8, 30.9, 26.4,
27.3, 29.4, 23)
tg_supp <- c("VC", "VC", "VC", "VC", "VC", "VC", "VC", "VC", "VC", "VC",
"VC", "VC", "VC", "VC", "VC", "VC", "VC", "VC", "VC", "VC", "VC",
"VC", "VC", "VC", "VC", "VC", "VC", "VC", "VC", "VC", "OJ", "OJ",
"OJ", "OJ", "OJ", "OJ", "OJ", "OJ", "OJ", "OJ", "OJ", "OJ", "OJ",
"OJ", "OJ", "OJ", "OJ", "OJ", "OJ", "OJ", "OJ", "OJ", "OJ", "OJ",
"OJ", "OJ", "OJ", "OJ", "OJ", "OJ")
tg_dose <- c(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0.5, 0.5,
0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
pg_weight <- c(4.17, 5.58, 5.18, 6.11, 4.5, 4.61, 5.17, 4.53, 5.33, 5.14,
4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31,
5.12, 5.54, 5.5, 5.37, 5.29, 4.92, 6.15, 5.8, 5.26)
pg_group <- c("ctrl", "ctrl", "ctrl", "ctrl", "ctrl", "ctrl", "ctrl", "ctrl",
"ctrl", "ctrl", "trt1", "trt1", "trt1", "trt1", "trt1", "trt1",
"trt1", "trt1", "trt1", "trt1", "trt2", "trt2", "trt2", "trt2",
"trt2", "trt2", "trt2", "trt2", "trt2", "trt2")
wb_breaks <- c(26, 30, 54, 25, 70, 52, 51, 26, 67, 18, 21, 29, 17, 12, 18,
35, 30, 36, 36, 21, 24, 18, 10, 43, 28, 15, 26, 27, 14, 29, 19,
29, 31, 41, 20, 44, 42, 26, 19, 16, 39, 28, 21, 39, 29, 20, 21,
24, 17, 13, 15, 15, 16, 28)
wb_wool <- c("A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
"A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
"A", "A", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B",
"B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B",
"B", "B", "B")
wb_tension <- c("L", "L", "L", "L", "L", "L", "L", "L", "L", "M", "M", "M",
"M", "M", "M", "M", "M", "M", "H", "H", "H", "H", "H", "H", "H",
"H", "H", "L", "L", "L", "L", "L", "L", "L", "L", "L", "M", "M",
"M", "M", "M", "M", "M", "M", "M", "H", "H", "H", "H", "H", "H",
"H", "H", "H")
mt_mpg <- c(21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8,
16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5,
15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4)
mt_cyl <- c(6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4,
4, 8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 4)
mt_hp <- c(110, 110, 93, 110, 175, 105, 245, 62, 95, 123, 123, 180, 180,
180, 205, 215, 230, 66, 52, 65, 97, 150, 150, 245, 175, 66, 91,
113, 264, 175, 335, 109)
mt_wt <- c(2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44,
3.44, 4.07, 3.73, 3.78, 5.25, 5.424, 5.345, 2.2, 1.615, 1.835,
2.465, 3.52, 3.435, 3.84, 3.845, 1.935, 2.14, 1.513, 3.17, 2.77,
3.57, 2.78)
mt_am <- c(1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1)
aq_ozone <- c(41L, 36L, 12L, 18L, 28L, 23L, 19L, 8L, 7L, 16L, 11L, 14L, 18L,
14L, 34L, 6L, 30L, 11L, 1L, 11L, 4L, 32L, 23L, 45L, 115L, 37L,
29L, 71L, 39L, 23L, 21L, 37L, 20L, 12L, 13L, 135L, 49L, 32L,
64L, 40L, 77L, 97L, 97L, 85L, 10L, 27L, 7L, 48L, 35L, 61L, 79L,
63L, 16L, 80L, 108L, 20L, 52L, 82L, 50L, 64L, 59L, 39L, 9L, 16L,
78L, 35L, 66L, 122L, 89L, 110L, 44L, 28L, 65L, 22L, 59L, 23L,
31L, 44L, 21L, 9L, 45L, 168L, 73L, 76L, 118L, 84L, 85L, 96L,
78L, 73L, 91L, 47L, 32L, 20L, 23L, 21L, 24L, 44L, 21L, 28L, 9L,
13L, 46L, 18L, 13L, 24L, 16L, 13L, 23L, 36L, 7L, 14L, 30L, 14L,
18L, 20L)
aq_temp <- c(67L, 72L, 74L, 62L, 66L, 65L, 59L, 61L, 74L, 69L, 66L, 68L,
58L, 64L, 66L, 57L, 68L, 62L, 59L, 73L, 61L, 61L, 67L, 81L, 79L,
76L, 82L, 90L, 87L, 82L, 77L, 72L, 65L, 73L, 76L, 84L, 85L, 81L,
83L, 83L, 88L, 92L, 92L, 89L, 73L, 81L, 80L, 81L, 82L, 84L, 87L,
85L, 74L, 86L, 85L, 82L, 86L, 88L, 86L, 83L, 81L, 81L, 81L, 82L,
86L, 85L, 87L, 89L, 90L, 90L, 86L, 82L, 80L, 77L, 79L, 76L, 78L,
78L, 77L, 72L, 79L, 81L, 86L, 97L, 94L, 96L, 94L, 91L, 92L, 93L,
93L, 87L, 84L, 80L, 78L, 75L, 73L, 81L, 76L, 77L, 71L, 71L, 78L,
67L, 76L, 68L, 82L, 64L, 71L, 81L, 69L, 63L, 70L, 75L, 76L, 68L
)
ff_eruptions <- c(3.6, 1.8, 3.333, 2.283, 4.533, 2.883, 4.7, 3.6, 1.95, 4.35,
1.833, 3.917, 4.2, 1.75, 4.7, 2.167, 1.75, 4.8, 1.6, 4.25, 1.8,
1.75, 3.45, 3.067, 4.533, 3.6, 1.967, 4.083, 3.85, 4.433, 4.3,
4.467, 3.367, 4.033, 3.833, 2.017, 1.867, 4.833, 1.833, 4.783,
4.35, 1.883, 4.567, 1.75, 4.533, 3.317, 3.833, 2.1, 4.633, 2,
4.8, 4.716, 1.833, 4.833, 1.733, 4.883, 3.717, 1.667, 4.567,
4.317, 2.233, 4.5, 1.75, 4.8, 1.817, 4.4, 4.167, 4.7, 2.067,
4.7, 4.033, 1.967, 4.5, 4, 1.983, 5.067, 2.017, 4.567, 3.883,
3.6, 4.133, 4.333, 4.1, 2.633, 4.067, 4.933, 3.95, 4.517, 2.167,
4, 2.2, 4.333, 1.867, 4.817, 1.833, 4.3, 4.667, 3.75, 1.867,
4.9, 2.483, 4.367, 2.1, 4.5, 4.05, 1.867, 4.7, 1.783, 4.85, 3.683,
4.733, 2.3, 4.9, 4.417, 1.7, 4.633, 2.317, 4.6, 1.817, 4.417,
2.617, 4.067, 4.25, 1.967, 4.6, 3.767, 1.917, 4.5, 2.267, 4.65,
1.867, 4.167, 2.8, 4.333, 1.833, 4.383, 1.883, 4.933, 2.033,
3.733, 4.233, 2.233, 4.533, 4.817, 4.333, 1.983, 4.633, 2.017,
5.1, 1.8, 5.033, 4, 2.4, 4.6, 3.567, 4, 4.5, 4.083, 1.8, 3.967,
2.2, 4.15, 2, 3.833, 3.5, 4.583, 2.367, 5, 1.933, 4.617, 1.917,
2.083, 4.583, 3.333, 4.167, 4.333, 4.5, 2.417, 4, 4.167, 1.883,
4.583, 4.25, 3.767, 2.033, 4.433, 4.083, 1.833, 4.417, 2.183,
4.8, 1.833, 4.8, 4.1, 3.966, 4.233, 3.5, 4.366, 2.25, 4.667,
2.1, 4.35, 4.133, 1.867, 4.6, 1.783, 4.367, 3.85, 1.933, 4.5,
2.383, 4.7, 1.867, 3.833, 3.417, 4.233, 2.4, 4.8, 2, 4.15, 1.867,
4.267, 1.75, 4.483, 4, 4.117, 4.083, 4.267, 3.917, 4.55, 4.083,
2.417, 4.183, 2.217, 4.45, 1.883, 1.85, 4.283, 3.95, 2.333, 4.15,
2.35, 4.933, 2.9, 4.583, 3.833, 2.083, 4.367, 2.133, 4.35, 2.2,
4.45, 3.567, 4.5, 4.15, 3.817, 3.917, 4.45, 2, 4.283, 4.767,
4.533, 1.85, 4.25, 1.983, 2.25, 4.75, 4.117, 2.15, 4.417, 1.817,
4.467)
ff_waiting <- c(79, 54, 74, 62, 85, 55, 88, 85, 51, 85, 54, 84, 78, 47, 83,
52, 62, 84, 52, 79, 51, 47, 78, 69, 74, 83, 55, 76, 78, 79, 73,
77, 66, 80, 74, 52, 48, 80, 59, 90, 80, 58, 84, 58, 73, 83, 64,
53, 82, 59, 75, 90, 54, 80, 54, 83, 71, 64, 77, 81, 59, 84, 48,
82, 60, 92, 78, 78, 65, 73, 82, 56, 79, 71, 62, 76, 60, 78, 76,
83, 75, 82, 70, 65, 73, 88, 76, 80, 48, 86, 60, 90, 50, 78, 63,
72, 84, 75, 51, 82, 62, 88, 49, 83, 81, 47, 84, 52, 86, 81, 75,
59, 89, 79, 59, 81, 50, 85, 59, 87, 53, 69, 77, 56, 88, 81, 45,
82, 55, 90, 45, 83, 56, 89, 46, 82, 51, 86, 53, 79, 81, 60, 82,
77, 76, 59, 80, 49, 96, 53, 77, 77, 65, 81, 71, 70, 81, 93, 53,
89, 45, 86, 58, 78, 66, 76, 63, 88, 52, 93, 49, 57, 77, 68, 81,
81, 73, 50, 85, 74, 55, 77, 83, 83, 51, 78, 84, 46, 83, 55, 81,
57, 76, 84, 77, 81, 87, 77, 51, 78, 60, 82, 91, 53, 78, 46, 77,
84, 49, 83, 71, 80, 49, 75, 64, 76, 53, 94, 55, 76, 50, 82, 54,
75, 78, 79, 78, 78, 70, 79, 70, 54, 86, 50, 90, 54, 54, 77, 79,
64, 75, 47, 86, 63, 85, 82, 57, 82, 67, 74, 54, 83, 73, 73, 88,
80, 71, 83, 56, 79, 78, 84, 58, 83, 43, 60, 75, 81, 46, 90, 46,
74)
is_count <- c(10, 7, 20, 14, 14, 12, 10, 23, 17, 20, 14, 13, 11, 17, 21,
11, 16, 14, 17, 17, 19, 21, 7, 13, 0, 1, 7, 2, 3, 1, 2, 1, 3,
0, 1, 4, 3, 5, 12, 6, 4, 3, 5, 5, 5, 5, 2, 4, 3, 5, 3, 5, 3,
6, 1, 1, 3, 2, 6, 4, 11, 9, 15, 22, 15, 16, 13, 10, 26, 26, 24,
13)
is_spray <- c("A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
"B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "C",
"C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "D", "D",
"D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "E", "E", "E",
"E", "E", "E", "E", "E", "E", "E", "E", "E", "F", "F", "F", "F",
"F", "F", "F", "F", "F", "F", "F", "F")
cw_weight <- c(179, 160, 136, 227, 217, 168, 108, 124, 143, 140, 309, 229,
181, 141, 260, 203, 148, 169, 213, 257, 244, 271, 243, 230, 248,
327, 329, 250, 193, 271, 316, 267, 199, 171, 158, 248, 423, 340,
392, 339, 341, 226, 320, 295, 334, 322, 297, 318, 325, 257, 303,
315, 380, 153, 263, 242, 206, 344, 258, 368, 390, 379, 260, 404,
318, 352, 359, 216, 222, 283, 332)
cw_feed <- c("horsebean", "horsebean", "horsebean", "horsebean", "horsebean",
"horsebean", "horsebean", "horsebean", "horsebean", "horsebean",
"linseed", "linseed", "linseed", "linseed", "linseed", "linseed",
"linseed", "linseed", "linseed", "linseed", "linseed", "linseed",
"soybean", "soybean", "soybean", "soybean", "soybean", "soybean",
"soybean", "soybean", "soybean", "soybean", "soybean", "soybean",
"soybean", "soybean", "sunflower", "sunflower", "sunflower",
"sunflower", "sunflower", "sunflower", "sunflower", "sunflower",
"sunflower", "sunflower", "sunflower", "sunflower", "meatmeal",
"meatmeal", "meatmeal", "meatmeal", "meatmeal", "meatmeal", "meatmeal",
"meatmeal", "meatmeal", "meatmeal", "meatmeal", "casein", "casein",
"casein", "casein", "casein", "casein", "casein", "casein", "casein",
"casein", "casein", "casein")
sl_air <- c(80, 80, 75, 62, 62, 62, 62, 62, 58, 58, 58, 58, 58, 58, 50,
50, 50, 50, 50, 56, 70)
sl_water <- c(27, 27, 25, 24, 22, 23, 24, 24, 23, 18, 18, 17, 18, 19, 18,
18, 19, 19, 20, 20, 20)
sl_acid <- c(89, 88, 90, 87, 87, 87, 93, 93, 87, 80, 89, 88, 82, 93, 89,
86, 72, 79, 80, 82, 91)
sl_loss <- c(42, 37, 37, 28, 18, 18, 19, 20, 15, 14, 14, 13, 11, 12, 8,
7, 8, 8, 9, 15, 15)
es_agegp <- c(1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L,
2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 3L,
3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 4L,
4L, 4L, 4L, 4L, 4L, 4L, 4L, 4L, 4L, 4L, 4L, 4L, 4L, 4L, 4L, 5L,
5L, 5L, 5L, 5L, 5L, 5L, 5L, 5L, 5L, 5L, 5L, 5L, 5L, 5L, 6L, 6L,
6L, 6L, 6L, 6L, 6L, 6L, 6L, 6L, 6L)
es_alcgp <- c(1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L, 3L, 3L, 3L, 4L, 4L, 4L, 4L,
1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L, 3L, 3L, 3L, 3L, 4L, 4L, 4L, 1L,
1L, 1L, 1L, 2L, 2L, 2L, 2L, 3L, 3L, 3L, 3L, 4L, 4L, 4L, 4L, 1L,
1L, 1L, 1L, 2L, 2L, 2L, 2L, 3L, 3L, 3L, 3L, 4L, 4L, 4L, 4L, 1L,
1L, 1L, 1L, 2L, 2L, 2L, 3L, 3L, 3L, 3L, 4L, 4L, 4L, 4L, 1L, 1L,
1L, 2L, 2L, 2L, 2L, 3L, 3L, 4L, 4L)
es_tobgp <- c(1L, 2L, 3L, 4L, 1L, 2L, 3L, 4L, 1L, 2L, 4L, 1L, 2L, 3L, 4L,
1L, 2L, 3L, 4L, 1L, 2L, 3L, 4L, 1L, 2L, 3L, 4L, 1L, 2L, 3L, 1L,
2L, 3L, 4L, 1L, 2L, 3L, 4L, 1L, 2L, 3L, 4L, 1L, 2L, 3L, 4L, 1L,
2L, 3L, 4L, 1L, 2L, 3L, 4L, 1L, 2L, 3L, 4L, 1L, 2L, 3L, 4L, 1L,
2L, 3L, 4L, 1L, 2L, 3L, 1L, 2L, 3L, 4L, 1L, 2L, 3L, 4L, 1L, 2L,
4L, 1L, 2L, 3L, 4L, 1L, 2L, 1L, 2L)
es_ncases <- c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
3, 1, 0, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 0, 6, 4, 5, 5, 3, 6, 1,
2, 4, 3, 2, 4, 2, 3, 3, 4, 9, 6, 4, 3, 9, 8, 3, 4, 5, 6, 2, 5,
5, 4, 2, 0, 17, 3, 5, 6, 4, 2, 1, 3, 1, 1, 1, 1, 2, 1, 2, 1,
0, 1, 1, 1, 2, 1)
es_ncontrols <- c(40, 10, 6, 5, 27, 7, 4, 7, 2, 1, 2, 1, 0, 1, 2, 60, 13, 7,
8, 35, 20, 13, 8, 11, 6, 2, 1, 1, 3, 2, 45, 18, 10, 4, 32, 17,
10, 2, 13, 8, 4, 2, 0, 1, 1, 0, 47, 19, 9, 2, 31, 15, 13, 3,
9, 7, 3, 0, 5, 1, 1, 1, 43, 10, 5, 2, 17, 7, 4, 7, 8, 1, 0, 1,
1, 0, 0, 17, 4, 2, 3, 2, 3, 0, 0, 0, 0, 0)
lh_series <- c(2.4, 2.4, 2.4, 2.2, 2.1, 1.5, 2.3, 2.3, 2.5, 2, 1.9, 1.7, 2.2,
1.8, 3.2, 3.2, 2.7, 2.2, 2.2, 1.9, 1.9, 1.8, 2.7, 3, 2.3, 2,
2, 2.9, 2.9, 2.7, 2.7, 2.3, 2.6, 2.4, 1.8, 1.7, 1.5, 1.4, 2.1,
3.3, 3.5, 3.5, 3.1, 2.6, 2.1, 3.4, 3, 2.9)
lake_huron <- c(580.38, 581.86, 580.97, 580.8, 579.79, 580.39, 580.42, 580.82,
581.4, 581.32, 581.44, 581.68, 581.17, 580.53, 580.01, 579.91,
579.14, 579.16, 579.55, 579.67, 578.44, 578.24, 579.1, 579.09,
579.35, 578.82, 579.32, 579.01, 579, 579.8, 579.83, 579.72, 579.89,
580.01, 579.37, 578.69, 578.19, 578.67, 579.55, 578.92, 578.09,
579.37, 580.13, 580.14, 579.51, 579.24, 578.66, 578.86, 578.05,
577.79, 576.75, 576.75, 577.82, 578.64, 580.58, 579.48, 577.38,
576.9, 576.94, 576.24, 576.84, 576.85, 576.9, 577.79, 578.18,
577.51, 577.23, 578.42, 579.61, 579.05, 579.26, 579.22, 579.38,
579.1, 577.95, 578.12, 579.75, 580.85, 580.41, 579.96, 579.61,
578.76, 578.18, 577.21, 577.13, 579.1, 578.25, 577.91, 576.89,
575.96, 576.8, 577.68, 578.38, 578.52, 579.74, 579.31, 579.89,
579.96)
pur_conc <- c(0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56,
1.1, 1.1)
pur_rate <- c(76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200)

lh <- lh_series
LakeHuron <- lake_huron

# Small published examples (R help pages; Hollander, Wolfe and Chicken).
depression_x <- c(1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30)
depression_y <- c(0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29)
permeability_x <- c(0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46)
permeability_y <- c(1.15, 0.88, 0.90, 0.74, 1.21)
rounding_times <- matrix(c(
  5.40, 5.50, 5.55, 5.85, 5.70, 5.75, 5.20, 5.60, 5.50, 5.55, 5.50, 5.40,
  5.90, 5.85, 5.70, 5.45, 5.55, 5.60, 5.40, 5.40, 5.35, 5.45, 5.50, 5.35,
  5.25, 5.15, 5.00, 5.85, 5.80, 5.70, 5.25, 5.20, 5.10, 5.65, 5.55, 5.45,
  5.60, 5.35, 5.45, 5.05, 5.00, 4.95, 5.50, 5.50, 5.40, 5.45, 5.55, 5.50,
  5.55, 5.55, 5.35, 5.45, 5.50, 5.55, 5.50, 5.45, 5.25, 5.65, 5.60, 5.40,
  5.70, 5.65, 5.55, 6.30, 6.30, 6.25), nrow = 22, byrow = TRUE)
tea_tasting <- matrix(c(3, 1, 1, 3), nrow = 2)
convictions <- matrix(c(2, 10, 15, 3), nrow = 2)
job_satisfaction <- matrix(c(1, 2, 1, 0, 3, 3, 6, 1, 10, 10, 14, 9, 6, 7, 12, 11), nrow = 4)
party_by_gender <- rbind(c(762, 327, 468), c(484, 239, 477))
ucb_admissions <- rbind(c(1198, 557), c(1493, 1278))
performance <- matrix(c(794, 86, 150, 570), nrow = 2)
aml_time <- c(9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161, 5, 5, 8, 8, 12, 16,
              23, 27, 30, 33, 43, 45)
aml_status <- c(1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                1, 1, 1)
aml_x <- rep(c("Maintained", "Nonmaintained"), c(11, 12))
portpirie_sea_level <- c(
  4.03, 3.83, 3.65, 3.88, 4.01, 4.08, 4.18, 3.8, 4.36, 3.96, 3.98, 4.69,
  3.85, 3.96, 3.85, 3.93, 3.75, 3.63, 3.57, 4.25, 3.97, 4.05, 4.24, 4.22,
  3.73, 4.37, 4.06, 3.71, 3.96, 4.06, 4.55, 3.79, 3.89, 4.11, 3.85, 3.86,
  3.86, 4.21, 4.01, 4.11, 4.24, 3.96, 4.21, 3.74, 3.85, 3.88, 3.66, 4.11,
  3.71, 4.18, 3.9, 3.78, 3.91, 3.72, 4, 3.66, 3.62, 4.33, 4.55, 3.75, 4.08,
  3.9, 3.88, 3.94, 4.33)
portpirie <- portpirie_sea_level
p_values <- c(0.0001, 0.004, 0.019, 0.023, 0.04, 0.041, 0.2, 0.62)

# ------------------------------------------------------------- data frames

sleep <- data.frame(extra = sleep_extra, group = factor(sleep_group),
                    ID = factor(sleep_id))
ToothGrowth <- data.frame(len = tg_len, supp = factor(tg_supp, levels = c("OJ", "VC")),
                          dose = tg_dose)
PlantGrowth <- data.frame(weight = pg_weight, group = factor(pg_group))
warpbreaks <- data.frame(breaks = wb_breaks, wool = factor(wb_wool),
                         tension = factor(wb_tension, levels = c("L", "M", "H")))
mtcars <- data.frame(mpg = mt_mpg, cyl = mt_cyl, hp = mt_hp, wt = mt_wt, am = mt_am)
airquality <- data.frame(Ozone = aq_ozone, Temp = aq_temp)
faithful <- data.frame(eruptions = ff_eruptions, waiting = ff_waiting)
InsectSprays <- data.frame(count = is_count, spray = factor(is_spray))
chickwts <- data.frame(weight = cw_weight, feed = factor(cw_feed))
stackloss <- data.frame(Air.Flow = sl_air, Water.Temp = sl_water,
                        Acid.Conc. = sl_acid, stack.loss = sl_loss)
esoph <- data.frame(agegp = es_agegp, alcgp = es_alcgp, tobgp = es_tobgp,
                    ncases = es_ncases, ncontrols = es_ncontrols)
aml <- data.frame(time = aml_time, status = aml_status, x = factor(aml_x))
Puromycin <- data.frame(conc = pur_conc, rate = pur_rate)

# The literal copies must equal the datasets shipped with R (airquality is
# restricted to rows with both Ozone and Temp, esoph's ordered factors are
# stored as integer scores, and Puromycin to the treated state).
same_column <- function(a, b) {
  if (is.factor(b)) identical(as.character(a), as.character(b)) &&
    identical(levels(a), levels(b))
  else isTRUE(all(as.numeric(a) == as.numeric(b))) && length(a) == length(b)
}
check_copy <- function(name, copy, original) {
  for (column in names(copy)) {
    if (!same_column(copy[[column]], original[[column]])) {
      stop("literal copy of ", name, "$", column, " differs from R's dataset")
    }
  }
}
check_copy("sleep", sleep, datasets::sleep)
check_copy("ToothGrowth", ToothGrowth, datasets::ToothGrowth)
check_copy("PlantGrowth", PlantGrowth, datasets::PlantGrowth)
check_copy("warpbreaks", warpbreaks, datasets::warpbreaks)
check_copy("mtcars", mtcars, datasets::mtcars)
complete <- stats::complete.cases(datasets::airquality[, c("Ozone", "Temp")])
check_copy("airquality", airquality, datasets::airquality[complete, ])
check_copy("faithful", faithful, datasets::faithful)
check_copy("InsectSprays", InsectSprays, datasets::InsectSprays)
check_copy("chickwts", chickwts, datasets::chickwts)
check_copy("stackloss", stackloss, datasets::stackloss)
esoph_scores <- data.frame(
  agegp = as.integer(datasets::esoph$agegp), alcgp = as.integer(datasets::esoph$alcgp),
  tobgp = as.integer(datasets::esoph$tobgp), ncases = datasets::esoph$ncases,
  ncontrols = datasets::esoph$ncontrols)
check_copy("esoph", esoph, esoph_scores)
check_copy("aml", aml, survival::aml)
check_copy("Puromycin", Puromycin, datasets::Puromycin[datasets::Puromycin$state == "treated", ])
stopifnot(same_column(lh, datasets::lh), same_column(LakeHuron, datasets::LakeHuron))
stopifnot(same_column(ucb_admissions, apply(datasets::UCBAdmissions, c(1, 2), sum)))
local({
  e <- new.env()
  utils::data("portpirie", package = "ismev", envir = e)
  stopifnot(same_column(portpirie_sea_level, e$portpirie$SeaLevel))
})

# ------------------------------------------------------------------ helpers
# Extract the numeric parts of an "htest" object under probstat-style names.
htest_values <- function(r) {
  out <- list(statistic = unname(r$statistic))
  if (!is.null(r$parameter)) out$df <- unname(r$parameter)
  out$p_value <- r$p.value
  if (!is.null(r$conf.int)) out$conf_int <- as.numeric(r$conf.int)
  out
}

# Levene-type statistic: one-way ANOVA F of absolute deviations from the
# group centre (mean for Levene 1960, median for Brown and Forsythe 1974),
# the base-R equivalent of car::leveneTest.
abs_deviation_anova <- function(y, g, centre) {
  d <- abs(y - ave(y, g, FUN = centre))
  a <- anova(lm(d ~ g))
  list(statistic = a[["F value"]][1], df = a[["Df"]], p_value = a[["Pr(>F)"]][1])
}

# =========================================================== distributions

add_law("dist.norm.pdf", "Normal", list(mu = 1.5, sigma = 2), "pdf",
        "dnorm", "mean = 1.5, sd = 2", c(-30, -3, 0, 1.5, 4, 25))
add_law("dist.norm.cdf", "Normal", list(mu = 0, sigma = 1), "cdf",
        "pnorm", "mean = 0, sd = 1", c(-38.5, -20, -5, -1.96, 0, 0.3, 3))
add_law("dist.norm.sf", "Normal", list(mu = 0, sigma = 1), "sf",
        "pnorm", "mean = 0, sd = 1", c(-3, 0, 5, 20, 38), tail = ", lower.tail = FALSE")
add_law("dist.norm.quantile", "Normal", list(mu = 0, sigma = 1), "quantile",
        "qnorm", "mean = 0, sd = 1", c(1e-300, 1e-20, 1e-5, 0.025, 0.5, 0.8, 0.999999))
add_law("dist.t.pdf", "StudentT", list(nu = 4), "pdf",
        "dt", "df = 4", c(-10, -1, 0, 2, 50))
add_law("dist.t.cdf", "StudentT", list(nu = 3.5), "cdf",
        "pt", "df = 3.5", c(-50, -4, -1, 0, 0.5, 2.5, 30), reason = same_special)
add_law("dist.t.quantile", "StudentT", list(nu = 7), "quantile",
        "qt", "df = 7", c(1e-10, 0.001, 0.025, 0.5, 0.9, 0.9999), reason = same_special)
add_law("dist.chisq.cdf", "ChiSquare", list(nu = 5), "cdf",
        "pchisq", "df = 5", c(1e-4, 0.5, 3, 10, 40), reason = same_special)
add_law("dist.chisq.sf", "ChiSquare", list(nu = 5), "sf",
        "pchisq", "df = 5", c(50, 100, 200), tail = ", lower.tail = FALSE",
        reason = same_special)
add_law("dist.chisq.quantile", "ChiSquare", list(nu = 2.5), "quantile",
        "qchisq", "df = 2.5", c(1e-8, 0.01, 0.5, 0.95, 0.999999), reason = same_special)
add_law("dist.f.cdf", "FisherF", list(nu1 = 4, nu2 = 12), "cdf",
        "pf", "df1 = 4, df2 = 12", c(0.01, 0.5, 1, 3, 20), reason = same_special)
add_law("dist.f.quantile", "FisherF", list(nu1 = 3, nu2 = 17), "quantile",
        "qf", "df1 = 3, df2 = 17", c(0.001, 0.05, 0.5, 0.95, 0.999), reason = same_special)
add_law("dist.gamma.pdf", "Gamma", list(shape = 2.5, rate = 1.5), "pdf",
        "dgamma", "shape = 2.5, rate = 1.5", c(0.01, 0.5, 1, 3, 12))
add_law("dist.gamma.cdf", "Gamma", list(shape = 2.5, rate = 1.5), "cdf",
        "pgamma", "shape = 2.5, rate = 1.5", c(1e-3, 0.2, 1, 4, 15), reason = same_special)
add_law("dist.gamma.quantile", "Gamma", list(shape = 0.4, rate = 2), "quantile",
        "qgamma", "shape = 0.4, rate = 2", c(1e-6, 0.05, 0.5, 0.95, 1 - 1e-9),
        reason = same_special)
add_law("dist.beta.pdf", "Beta", list(alpha = 2.5, beta = 0.7), "pdf",
        "dbeta", "shape1 = 2.5, shape2 = 0.7", c(0.01, 0.3, 0.7, 0.99))
add_law("dist.beta.cdf", "Beta", list(alpha = 2.5, beta = 0.7), "cdf",
        "pbeta", "shape1 = 2.5, shape2 = 0.7", c(1e-6, 0.1, 0.5, 0.9, 0.999999),
        reason = same_special)
add_law("dist.beta.quantile", "Beta", list(alpha = 0.5, beta = 3), "quantile",
        "qbeta", "shape1 = 0.5, shape2 = 3", c(1e-8, 0.025, 0.5, 0.975, 0.99999),
        reason = same_special)
add_law("dist.binom.pmf", "Binomial", list(n = 40, p = 0.3), "pmf",
        "dbinom", "size = 40, prob = 0.3", c(0, 3, 10, 25, 40))
add_law("dist.binom.cdf", "Binomial", list(n = 40, p = 0.3), "cdf",
        "pbinom", "size = 40, prob = 0.3", c(0, 5, 12, 20, 39), reason = same_special)
add_law("dist.binom.quantile", "Binomial", list(n = 40, p = 0.3), "quantile",
        "qbinom", "size = 40, prob = 0.3", c(0.001, 0.1, 0.5, 0.9, 0.999),
        relative = 0, reason = "Discrete quantiles are integers and must be equal.")
add_law("dist.pois.pmf", "Poisson", list(lam = 7.5), "pmf",
        "dpois", "lambda = 7.5", c(0, 1, 5, 20, 60))
add_law("dist.pois.cdf", "Poisson", list(lam = 7.5), "cdf",
        "ppois", "lambda = 7.5", c(0, 3, 7, 15, 30), reason = same_special)
add_law("dist.pois.quantile", "Poisson", list(lam = 7.5), "quantile",
        "qpois", "lambda = 7.5", c(0.001, 0.25, 0.5, 0.75, 0.9999),
        relative = 0, reason = "Discrete quantiles are integers and must be equal.")
add_law("dist.nbinom.pmf", "NegativeBinomial", list(r = 3, p = 0.4), "pmf",
        "dnbinom", "size = 3, prob = 0.4", c(0, 2, 5, 15, 40))
add_law("dist.nbinom.cdf", "NegativeBinomial", list(r = 3, p = 0.4), "cdf",
        "pnbinom", "size = 3, prob = 0.4", c(0, 2, 5, 15, 40), reason = same_special)
add_law("dist.weibull.cdf", "Weibull", list(shape = 1.7, scale = 2), "cdf",
        "pweibull", "shape = 1.7, scale = 2", c(0.01, 0.5, 1, 2, 5, 12),
        reason = same_closed_form)
add_law("dist.weibull.quantile", "Weibull", list(shape = 1.7, scale = 2), "quantile",
        "qweibull", "shape = 1.7, scale = 2", c(1e-9, 0.1, 0.5, 0.9, 0.999999),
        reason = same_closed_form)
add_law("dist.lnorm.pdf", "LogNormal", list(mu = 0.3, sigma = 0.8), "pdf",
        "dlnorm", "meanlog = 0.3, sdlog = 0.8", c(0.01, 0.5, 1, 3, 50))
add_law("dist.lnorm.cdf", "LogNormal", list(mu = 0.3, sigma = 0.8), "cdf",
        "plnorm", "meanlog = 0.3, sdlog = 0.8", c(0.01, 0.5, 1, 3, 50))
add_law("dist.lnorm.quantile", "LogNormal", list(mu = 0.3, sigma = 0.8), "quantile",
        "qlnorm", "meanlog = 0.3, sdlog = 0.8", c(1e-12, 0.05, 0.5, 0.95, 0.9999999))
add_law("dist.exp.cdf", "Exponential", list(rate = 0.7), "cdf",
        "pexp", "rate = 0.7", c(1e-8, 0.1, 1, 10, 50), reason = same_closed_form)
add_law("dist.exp.quantile", "Exponential", list(rate = 0.7), "quantile",
        "qexp", "rate = 0.7", c(1e-12, 0.1, 0.5, 0.9, 1 - 1e-12), reason = same_closed_form)
add_law("dist.unif.cdf", "Uniform", list(lower = 0.5, upper = 4), "cdf",
        "punif", "min = 0.5, max = 4", c(-1, 0.5, 2, 3.9, 5), reason = same_closed_form)
add_law("dist.unif.quantile", "Uniform", list(lower = 0.5, upper = 4), "quantile",
        "qunif", "min = 0.5, max = 4", c(0, 0.1, 0.5, 0.999, 1), reason = same_closed_form)
add_law("dist.hyper.pmf", "Hypergeometric",
        list(population = 50, successes = 30, draws = 12), "pmf",
        "dhyper", "m = 30, n = 20, k = 12", c(0, 2, 5, 8, 12))
add_law("dist.hyper.cdf", "Hypergeometric",
        list(population = 50, successes = 30, draws = 12), "cdf",
        "phyper", "m = 30, n = 20, k = 12", c(0, 3, 6, 9, 11))
add_law("dist.geom.pmf", "Geometric", list(p = 0.3), "pmf",
        "dgeom", "prob = 0.3", c(0, 1, 5, 20, 80), reason = same_closed_form)
add_law("dist.geom.cdf", "Geometric", list(p = 0.3), "cdf",
        "pgeom", "prob = 0.3", c(0, 1, 5, 20, 50), reason = same_closed_form)
add_law("dist.logis.cdf", "Logistic", list(mu = 1, s = 2), "cdf",
        "plogis", "location = 1, scale = 2", c(-40, -3, 0, 1, 40), reason = same_closed_form)
add_law("dist.logis.quantile", "Logistic", list(mu = 1, s = 2), "quantile",
        "qlogis", "location = 1, scale = 2", c(1e-10, 0.1, 0.5, 0.75, 0.9999999),
        reason = same_closed_form)
add_law("dist.cauchy.cdf", "Cauchy", list(loc = 0.5, scale = 2), "cdf",
        "pcauchy", "location = 0.5, scale = 2", c(-1e6, -5, 0, 1, 1e3),
        reason = same_closed_form)
add_law("dist.cauchy.quantile", "Cauchy", list(loc = 0.5, scale = 2), "quantile",
        "qcauchy", "location = 0.5, scale = 2", c(1e-8, 0.1, 0.5, 0.75, 0.999999),
        reason = same_closed_form)

# Noncentral t and the studentized range: R evaluates both by series or
# quadrature with stated error bounds.
add_case("dist.noncentral_t.cdf", "distributions", "probstat.inference.noncentral_t_cdf",
         "list(values = pt(c(-1, 0.5, 2, 4), df = 12, ncp = 1.5))",
         0, 1e-11,
         "R's pnt (AS 243, Lenth 1989) sums its series to errmax = 1e-12 absolute; probstat integrates by adaptive quadrature.",
         probstat = list(df = 12, noncentrality = 1.5, points = I(c(-1, 0.5, 2, 4))))
add_case("dist.studentized_range.cdf", "distributions",
         "probstat.inference.studentized_range_cdf",
         "list(values = ptukey(c(1, 2.5, 3.5, 5), nmeans = 4, df = 20))",
         0, 1e-10,
         "R's ptukey (Copenhaver and Holland 1988) integrates with fixed-order Gauss-Legendre rules and no error estimate; probstat's adaptive quadrature is accurate to about 1e-12. The bound allows 1e-10 absolute.",
         probstat = list(groups = 4, df = 20, points = I(c(1, 2.5, 3.5, 5))))

# ============================================================== descriptive

quantile_probs <- c(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
# All nine Hyndman-Fan types of probstat.statistics.quantile.
for (type in 1:9) {
  add_case(sprintf("descriptive.quantile.type%d", type), "descriptive",
           "probstat.statistics.quantiles",
           sprintf("list(values = quantile(sleep$extra, probs = c(%s), type = %d))",
                   num_list(quantile_probs), type),
           1e-12, 1e-14,
           "Hyndman-Fan definition evaluated as the same order-statistic interpolation on both sides.",
           inputs = "sleep",
           probstat = list(method = type, probabilities = I(quantile_probs)))
}
# Levels at which n p is not an integer for n = 272, so the step types do not
# depend on how the product n p rounds (see the note).
quantile_levels_faithful <- c(0.01, 0.05, 1 / 3, 0.37, 0.5, 0.63, 0.95, 0.99)
add_case("descriptive.quantile.all_types.faithful", "descriptive",
         "probstat.statistics.quantiles",
         sprintf("setNames(lapply(1:9, function(t) quantile(faithful$eruptions, probs = c(%s), type = t)), paste0(\"type\", 1:9))",
                 num_list(quantile_levels_faithful)),
         1e-12, 1e-14,
         "Hyndman-Fan definition evaluated as the same order-statistic interpolation on both sides.",
         inputs = "faithful",
         note = "Where n p is an integer in exact arithmetic but the floating-point product is not (n = 100, p = 0.07), R compares the rounded product and its step types 1-3 can return the next order statistic; probstat forms the position from the level as written. The levels here avoid that case.",
         probstat = list(probabilities = I(quantile_levels_faithful)))
add_case("descriptive.variance", "descriptive", "probstat.statistics.variance",
         "list(variance = var(mtcars$mpg), standard_deviation = sd(mtcars$mpg))",
         1e-12, 0, same_closed_form, inputs = "mtcars",
         note = "probstat uses ddof = 1 for the n - 1 divisor of var() and sd().")
add_case("descriptive.mad", "descriptive", "probstat.statistics.median_absolute_deviation",
         "list(raw = mad(faithful$eruptions, constant = 1), normal = mad(faithful$eruptions, constant = 1 / qnorm(0.75)))",
         1e-12, 0, same_closed_form, inputs = "faithful",
         note = "R's default constant 1.4826 is the four-digit rounding of 1/qnorm(0.75) = 1.482602...; probstat uses the unrounded constant, so the R call passes it explicitly.")
add_case("descriptive.iqr", "descriptive", "probstat.statistics.interquartile_range",
         "list(iqr = IQR(faithful$waiting))", 1e-12, 0, same_closed_form,
         inputs = "faithful")
add_case("descriptive.trimmed_mean", "descriptive", "probstat.statistics.trimmed_mean",
         "list(trimmed_mean = mean(chickwts$weight, trim = 0.1))", 1e-12, 0,
         same_closed_form, inputs = "chickwts")
add_case("descriptive.cor.pearson", "descriptive", "probstat.statistics.pearson_correlation",
         "list(correlation = cor(mtcars$mpg, mtcars$wt))", 1e-12, 0, same_closed_form,
         inputs = "mtcars")
add_case("descriptive.cor.spearman", "descriptive", "probstat.statistics.spearman_correlation",
         "list(correlation = cor(mtcars$mpg, mtcars$hp, method = \"spearman\"))", 1e-12, 0,
         "Pearson correlation of mid-ranks on both sides (hp and mpg contain ties).",
         inputs = "mtcars")
add_case("descriptive.cor.kendall", "descriptive", "probstat.modeling.kendall_tau",
         "list(correlation = cor(mtcars$mpg, mtcars$hp, method = \"kendall\"))", 1e-12, 0,
         "Both sides count concordant, discordant, and tied pairs exactly (tau-b with ties in both variables).",
         inputs = "mtcars")
add_case("descriptive.skewness.type1", "descriptive", "probstat.statistics.skewness",
         "list(skewness = e1071::skewness(faithful$eruptions, type = 1))", 1e-12, 0,
         same_closed_form, inputs = "faithful",
         note = "e1071 type 1 is the moment ratio g1 = m3 / m2^(3/2), probstat's definition.")
add_case("descriptive.kurtosis.type1", "descriptive", "probstat.statistics.excess_kurtosis",
         "list(excess_kurtosis = e1071::kurtosis(faithful$eruptions, type = 1))", 1e-12, 0,
         same_closed_form, inputs = "faithful",
         note = "e1071 type 1 is g2 = m4 / m2^2 - 3, probstat's definition.")
for (type in 2:3) {
  add_case(sprintf("descriptive.skewness.type%d", type), "descriptive",
           "probstat.statistics.skewness",
           sprintf("list(skewness = e1071::skewness(faithful$eruptions, type = %d))", type),
           1e-12, 0, same_closed_form, inputs = "faithful",
           note = "e1071 types 2 and 3 are Joanes and Gill's G1 and b1, probstat's skewness(type = 2) and skewness(type = 3).",
           probstat = list(type = type))
  add_case(sprintf("descriptive.kurtosis.type%d", type), "descriptive",
           "probstat.statistics.excess_kurtosis",
           sprintf("list(excess_kurtosis = e1071::kurtosis(faithful$eruptions, type = %d))", type),
           1e-12, 0, same_closed_form, inputs = "faithful",
           note = "e1071 types 2 and 3 are Joanes and Gill's G2 and b2, probstat's excess_kurtosis(type = 2) and excess_kurtosis(type = 3).",
           probstat = list(type = type))
}
add_case("descriptive.bandwidth.nrd0", "descriptive", "probstat.statistics.kde_bandwidth",
         "list(bandwidth = bw.nrd0(faithful$eruptions))", 1e-12, 0, same_closed_form,
         inputs = "faithful", probstat = list(rule = "silverman"))
add_case("descriptive.bandwidth.nrd", "descriptive", "probstat.statistics.kde_bandwidth",
         "list(bandwidth = bw.nrd(faithful$eruptions))", 1e-12, 0, same_closed_form,
         inputs = "faithful", probstat = list(rule = "nrd"))
add_case("descriptive.bandwidth.ucv", "descriptive", "probstat.statistics.select_kde_bandwidth",
         "list(bandwidth = bw.ucv(faithful$eruptions, nb = 10000000L, tol = 1e-10 * 1.144 * sd(faithful$eruptions) * length(faithful$eruptions)^(-1/5)))",
         1e-6, 0,
         "R bins every pairwise distance to a multiple of 1.01 range / nb before minimizing (at most about 1e-6 relative at nb = 1e7); probstat minimizes the criterion on exact distances.",
         inputs = "faithful", probstat = list(method = "ucv"))
add_case("descriptive.bandwidth.bcv", "descriptive", "probstat.statistics.select_kde_bandwidth",
         "list(bandwidth = bw.bcv(faithful$eruptions, nb = 10000000L, tol = 1e-10 * 1.144 * sd(faithful$eruptions) * length(faithful$eruptions)^(-1/5)))",
         1e-6, 0,
         "R bins every pairwise distance to a multiple of 1.01 range / nb before minimizing (at most about 1e-6 relative at nb = 1e7); probstat minimizes the criterion on exact distances.",
         inputs = "faithful", probstat = list(method = "bcv"))

# =================================================================== tests

add_case("tests.t.one_sample", "tests", "probstat.inference.one_sample_t_test",
         "c(htest_values(t.test(sleep$extra[sleep$group == \"2\"], mu = 1)), list(standard_error = t.test(sleep$extra[sleep$group == \"2\"], mu = 1)$stderr))",
         1e-10, 0, same_special, inputs = "sleep")
add_case("tests.t.welch", "tests", "probstat.inference.two_sample_t_test",
         "htest_values(t.test(ToothGrowth$len[ToothGrowth$supp == \"OJ\"], ToothGrowth$len[ToothGrowth$supp == \"VC\"]))",
         1e-10, 0, same_special, inputs = "ToothGrowth")
add_case("tests.t.pooled", "tests", "probstat.inference.two_sample_t_test",
         "htest_values(t.test(ToothGrowth$len[ToothGrowth$supp == \"OJ\"], ToothGrowth$len[ToothGrowth$supp == \"VC\"], var.equal = TRUE))",
         1e-10, 0, same_special, inputs = "ToothGrowth")
add_case("tests.t.paired", "tests", "probstat.inference.paired_t_test",
         "htest_values(t.test(sleep$extra[sleep$group == \"1\"], sleep$extra[sleep$group == \"2\"], paired = TRUE))",
         1e-10, 0, same_special, inputs = "sleep")
add_case("tests.t.power", "tests", "probstat.inference.t_test_power",
         "list(one_sample = power.t.test(n = 20, delta = 0.6, sd = 1, type = \"one.sample\", strict = TRUE)$power, two_sample = power.t.test(n = 15, delta = 0.8, sd = 1, type = \"two.sample\", strict = TRUE)$power)",
         0, 1e-11,
         "Both sides evaluate the noncentral t distribution function: R by the AS 243 series (errmax 1e-12 absolute), probstat by quadrature.",
         note = "strict = TRUE adds the lower rejection region, which probstat's exact two-sided power includes.")
add_case("tests.wilcoxon.signed_rank_exact", "tests", "probstat.inference.wilcoxon_signed_rank_test",
         "htest_values(wilcox.test(depression_x, depression_y, paired = TRUE, alternative = \"greater\"))",
         1e-12, 0, "Exact null distribution by counting on both sides.",
         inputs = c("depression_x", "depression_y"))
add_case("tests.wilcoxon.signed_rank_normal", "tests", "probstat.inference.wilcoxon_signed_rank_test",
         "htest_values(suppressWarnings(wilcox.test(sleep$extra[sleep$group == \"2\"], sleep$extra[sleep$group == \"1\"], paired = TRUE, exact = FALSE, correct = TRUE)))",
         1e-10, 0,
         "Same normal approximation (zeros dropped, tie-corrected variance, continuity correction) on both sides.",
         inputs = "sleep")
add_case("tests.wilcoxon.rank_sum_exact", "tests", "probstat.inference.mann_whitney_u_test",
         "htest_values(wilcox.test(permeability_x, permeability_y, alternative = \"greater\"))",
         1e-12, 0, "Exact null distribution by counting on both sides.",
         inputs = c("permeability_x", "permeability_y"),
         note = "R's W is the Mann-Whitney U of the first sample, probstat's statistic.")
add_case("tests.wilcoxon.rank_sum_normal", "tests", "probstat.inference.mann_whitney_u_test",
         "htest_values(wilcox.test(InsectSprays$count[InsectSprays$spray == \"C\"], InsectSprays$count[InsectSprays$spray == \"D\"], exact = FALSE, correct = TRUE))",
         1e-10, 0,
         "Same normal approximation (tie-corrected variance, continuity correction) on both sides.",
         inputs = "InsectSprays")
add_case("tests.binomial", "tests", "probstat.inference.binomial_test",
         "local({ r <- binom.test(682, 925, p = 0.75); list(p_value = r$p.value, conf_int = as.numeric(r$conf.int)) })",
         1e-10, 0,
         "Same minimum-likelihood two-sided sum (relative tolerance 1e-7 on the masses) and Clopper-Pearson interval from beta quantiles on both sides.")
add_case("tests.wilson_interval", "tests", "probstat.inference.wilson_score_interval",
         "list(conf_int = as.numeric(prop.test(15, 40, correct = FALSE)$conf.int))",
         1e-12, 0, same_closed_form,
         note = "prop.test without continuity correction reports the Wilson score interval.")
add_case("tests.fisher.greater", "tests", "probstat.inference.fisher_exact_test",
         "list(p_value = fisher.test(tea_tasting, alternative = \"greater\")$p.value)",
         1e-12, 0, "Exact hypergeometric tail on both sides.", inputs = "tea_tasting",
         note = "R reports the conditional maximum-likelihood odds ratio, probstat the sample odds ratio ad/bc; only the p-value is compared.")
add_case("tests.fisher.two_sided", "tests", "probstat.inference.fisher_exact_test",
         "list(p_value = fisher.test(convictions)$p.value)",
         1e-12, 0,
         "Exact hypergeometric sum over tables no more probable than the observed one (relative tolerance 1e-7) on both sides.",
         inputs = "convictions")
add_case("tests.fisher.rxc.job_satisfaction", "tests", "probstat.inference.fisher_exact_rxc_test",
         "list(p_value = fisher.test(job_satisfaction)$p.value)",
         1e-10, 0,
         "Both sides sum the exact multiple hypergeometric probabilities of the tables no more probable than the observed one; R's FEXACT and probstat's network sum add the floating-point terms in different orders (they agree to about 1e-12 here).",
         inputs = "job_satisfaction",
         note = "The 4 x 4 job-satisfaction table of the fisher.test help page (N = 96).")
add_case("tests.fisher.rxc.mtcars", "tests", "probstat.inference.fisher_exact_rxc_test",
         "list(p_value = fisher.test(table(mtcars$cyl, mtcars$am))$p.value)",
         1e-10, 0,
         "Both sides sum the exact multiple hypergeometric probabilities of the tables no more probable than the observed one, adding the floating-point terms in different orders.",
         inputs = "mtcars",
         note = "Cylinders by transmission, a 3 x 2 table with N = 32 built from the columns cyl and am.")
add_case("tests.fisher.rxc.large_counts", "tests", "probstat.inference.fisher_exact_rxc_test",
         "list(p_value = fisher.test(party_by_gender)$p.value)",
         1e-9, 0,
         "A 2 x 3 table with N = 2757. The log factorials reach 2e4, so each table probability carries a relative rounding error near 1e-12 on both sides; R's FEXACT result differs by 3e-10 relative from a full enumeration of the 476,499 tables in the tail (3.02747980552e-07), which probstat reproduces to 4e-12.",
         inputs = "party_by_gender")
add_case("tests.chisq.independence", "tests", "probstat.inference.chi_square_independence_test",
         "htest_values(chisq.test(party_by_gender))",
         1e-10, 0, same_special, inputs = "party_by_gender")
add_case("tests.chisq.yates", "tests", "probstat.inference.chi_square_independence_test",
         "htest_values(chisq.test(ucb_admissions, correct = TRUE))",
         1e-10, 0, same_special, inputs = "ucb_admissions")
add_case("tests.chisq.goodness_of_fit", "tests", "probstat.inference.chi_square_goodness_of_fit",
         "htest_values(suppressWarnings(chisq.test(c(89, 37, 30, 28, 2), p = c(0.40, 0.20, 0.20, 0.19, 0.01))))",
         1e-10, 0, same_special)
add_case("tests.mcnemar", "tests", "probstat.inference.mcnemar_test",
         "htest_values(mcnemar.test(performance))[c(\"statistic\", \"p_value\")]",
         1e-10, 0, same_special, inputs = "performance")
add_case("tests.ks.one_sample_exact", "tests", "probstat.inference.kolmogorov_smirnov_test",
         "htest_values(ks.test(depression_x, \"pnorm\", 1.5, 0.6))",
         1e-10, 0,
         "Same statistic; both sides compute the exact two-sided p-value for n = 9 (Marsaglia, Tsang and Wang 2003).",
         inputs = "depression_x")
add_case("tests.ks.one_sample_statistic", "tests", "probstat.inference.kolmogorov_smirnov_test",
         "list(statistic = unname(suppressWarnings(ks.test(faithful$eruptions, \"pnorm\", 3.5, 1.1))$statistic))",
         1e-12, 0, same_closed_form, inputs = "faithful",
         note = "R falls back to the asymptotic p-value for n = 272 with ties; probstat's p-value is exact, so only the statistic is compared.")
add_case("tests.ks.two_sample_exact", "tests",
         "probstat.inference.kolmogorov_smirnov_two_sample_test",
         "htest_values(ks.test(permeability_x, permeability_y))",
         1e-10, 0, "Same statistic and exact null distribution on both sides.",
         inputs = c("permeability_x", "permeability_y"))
add_case("tests.shapiro.small", "tests", "probstat.inference.shapiro_wilk_test",
         "htest_values(shapiro.test(PlantGrowth$weight))",
         1e-10, 0,
         "Both sides implement Royston's algorithm AS R94 (1995) with the same polynomial approximations.",
         inputs = "PlantGrowth")
add_case("tests.shapiro.large", "tests", "probstat.inference.shapiro_wilk_test",
         "htest_values(shapiro.test(faithful$waiting))",
         1e-10, 0,
         "Both sides implement Royston's algorithm AS R94 (1995) with the same polynomial approximations.",
         inputs = "faithful")
add_case("tests.kruskal.chickwts", "tests", "probstat.inference.kruskal_wallis_test",
         "htest_values(kruskal.test(weight ~ feed, data = chickwts))",
         1e-10, 0, same_special, inputs = "chickwts",
         note = "probstat is called with method = \"chi-square\", R's reference distribution.")
add_case("tests.kruskal.insectsprays", "tests", "probstat.inference.kruskal_wallis_test",
         "htest_values(kruskal.test(count ~ spray, data = InsectSprays))",
         1e-10, 0, same_special, inputs = "InsectSprays",
         note = "probstat is called with method = \"chi-square\", R's reference distribution.")
add_case("tests.friedman", "tests", "probstat.inference.friedman_test",
         "htest_values(friedman.test(rounding_times))",
         1e-10, 0, same_special, inputs = "rounding_times")
add_case("tests.bartlett.plantgrowth", "tests", "probstat.inference.bartlett_test",
         "htest_values(bartlett.test(weight ~ group, data = PlantGrowth))",
         1e-10, 0, same_special, inputs = "PlantGrowth")
add_case("tests.bartlett.chickwts", "tests", "probstat.inference.bartlett_test",
         "htest_values(bartlett.test(weight ~ feed, data = chickwts))",
         1e-10, 0, same_special, inputs = "chickwts")
add_case("tests.levene.mean", "tests", "probstat.inference.levene_test",
         "abs_deviation_anova(PlantGrowth$weight, PlantGrowth$group, mean)",
         1e-10, 0, same_special, inputs = "PlantGrowth",
         note = "car is not installed; Levene's statistic is the one-way ANOVA F of absolute deviations from the group means (car::leveneTest(center = mean)).")
add_case("tests.levene.median", "tests", "probstat.inference.brown_forsythe_test",
         "abs_deviation_anova(chickwts$weight, chickwts$feed, median)",
         1e-10, 0, same_special, inputs = "chickwts",
         note = "Brown-Forsythe statistic: one-way ANOVA F of absolute deviations from the group medians (car::leveneTest's default).")
add_case("tests.anova.one_way", "tests", "probstat.inference.one_way_anova",
         "htest_values(oneway.test(weight ~ group, data = PlantGrowth, var.equal = TRUE))",
         1e-10, 0, same_special, inputs = "PlantGrowth")
add_case("tests.anova.welch.plantgrowth", "tests", "probstat.inference.welch_anova",
         "htest_values(oneway.test(weight ~ group, data = PlantGrowth))",
         1e-10, 0, same_special, inputs = "PlantGrowth",
         note = "oneway.test defaults to var.equal = FALSE, Welch's test.")
add_case("tests.anova.welch.chickwts", "tests", "probstat.inference.welch_anova",
         "htest_values(oneway.test(weight ~ feed, data = chickwts, var.equal = FALSE))",
         1e-10, 0, same_special, inputs = "chickwts")
add_case("tests.fligner.insectsprays", "tests", "probstat.inference.fligner_killeen_test",
         "htest_values(fligner.test(count ~ spray, data = InsectSprays))",
         1e-10, 0,
         "Same mid-ranks, normal scores from the inverse normal CDF (Wichura's AS 241 on both sides), and chi-square tail through the regularized incomplete gamma function.",
         inputs = "InsectSprays",
         note = "The counts are heavily tied, which exercises the mid-ranks of the absolute deviations.")
add_case("tests.fligner.plantgrowth", "tests", "probstat.inference.fligner_killeen_test",
         "htest_values(fligner.test(weight ~ group, data = PlantGrowth))",
         1e-10, 0,
         "Same mid-ranks, normal scores from the inverse normal CDF (Wichura's AS 241 on both sides), and chi-square tail through the regularized incomplete gamma function.",
         inputs = "PlantGrowth")
add_case("tests.tukey.difference_and_p", "tests", "probstat.inference.tukey_hsd_test",
         "local({ tk <- TukeyHSD(aov(weight ~ group, data = PlantGrowth))$group; list(difference = tk[, \"diff\"], p_value = tk[, \"p adj\"]) })",
         0, 1e-10,
         "Differences are exact; the adjusted p-values come from R's ptukey (fixed-order Gauss-Legendre rules without an error estimate), so the bound is the 1e-10 absolute allowed for ptukey itself.",
         inputs = "PlantGrowth")
add_case("tests.tukey.interval", "tests", "probstat.inference.tukey_hsd_test",
         "local({ tk <- TukeyHSD(aov(weight ~ group, data = PlantGrowth))$group; list(lower = tk[, \"lwr\"], upper = tk[, \"upr\"]) })",
         0, 1e-10,
         "The bounds use R's qtukey, which inverts ptukey by a secant iteration, and probstat's quantile, which inverts its own distribution function by Brent's method to 1e-10; the bound is the 1e-10 absolute allowed for ptukey.",
         inputs = "PlantGrowth")
for (method in c("bonferroni", "holm", "BH", "BY")) {
  add_case(sprintf("tests.p_adjust.%s", tolower(method)), "tests",
           "probstat.inference.adjust_p_values",
           sprintf("list(adjusted = p.adjust(p_values, method = \"%s\"))", method),
           1e-12, 0, same_closed_form, inputs = "p_values",
           probstat = list(method = tolower(method)))
}

# ============================================================== regression

add_case("regression.lm.coefficients", "regression", "probstat.inference.fit_linear_model",
         "local({ s <- coef(summary(lm(mpg ~ wt + hp, data = mtcars))); list(coefficients = s[, 1], standard_errors = s[, 2], t_values = s[, 3], p_values = s[, 4]) })",
         1e-10, 0, "Both sides solve the same least-squares problem by orthogonal factorizations.",
         inputs = "mtcars")
add_case("regression.lm.summary", "regression", "probstat.inference.fit_linear_model",
         "local({ s <- summary(lm(mpg ~ wt + hp, data = mtcars)); list(r_squared = s$r.squared, adjusted_r_squared = s$adj.r.squared, f_statistic = s$fstatistic[1], sigma = s$sigma) })",
         1e-10, 0, "Closed forms of the same least-squares fit.", inputs = "mtcars")
add_case("regression.lm.information", "regression", "probstat.inference.fit_linear_model",
         "local({ fit <- lm(mpg ~ wt + hp, data = mtcars); list(log_likelihood = as.numeric(logLik(fit)), aic = AIC(fit), bic = BIC(fit)) })",
         1e-10, 0, "Closed forms of the same least-squares fit (p + 1 parameters).",
         inputs = "mtcars")
add_case("regression.lm.hatvalues", "regression", "probstat.inference.fit_linear_model",
         "list(leverage = hatvalues(lm(mpg ~ wt + hp, data = mtcars)))",
         1e-10, 0, "Closed forms of the same least-squares fit.", inputs = "mtcars")
add_case("regression.lm.residuals", "regression", "probstat.inference.fit_linear_model",
         "local({ fit <- lm(mpg ~ wt + hp, data = mtcars); list(standardized = rstandard(fit), studentized = rstudent(fit), cooks_distance = cooks.distance(fit)) })",
         1e-9, 0,
         "Closed forms of the same least-squares fit; the deleted residuals divide by small differences, so the relative tolerance allows one extra digit of rounding.",
         inputs = "mtcars")
add_case("regression.anova.sequential", "regression", "probstat.inference.linear_model_anova",
         "local({ a <- anova(lm(mpg ~ wt + factor(cyl), data = mtcars)); list(df = a$Df, sum_squares = a[[\"Sum Sq\"]], f_statistic = a[[\"F value\"]][1:2], p_value = a[[\"Pr(>F)\"]][1:2]) })",
         1e-10, 0, "Sequential (type I) sums of squares of the same least-squares fits.",
         inputs = "mtcars",
         note = "probstat is called with kind = \"I\"; the design is unbalanced, so type I and type II differ.")
add_case("regression.glm.binomial", "regression", "probstat.inference.fit_glm",
         "local({ fit <- glm(am ~ wt, family = binomial, data = mtcars, control = glm.control(epsilon = 1e-14, maxit = 100)); fit <- update(fit, start = coef(fit)); list(coefficients = coef(fit), standard_errors = sqrt(diag(vcov(fit))), deviance = deviance(fit), null_deviance = fit$null.deviance, aic = AIC(fit)) })",
         1e-9, 0,
         "Both sides run IRLS to convergence (R with epsilon = 1e-14) and evaluate the covariance at the solution: the R call refits from the converged coefficients because vcov() of a glm uses the working weights of the iteration before the final update (a 1e-8 relative effect on esoph).",
         inputs = "mtcars")
add_case("regression.glm.binomial_trials", "regression", "probstat.inference.fit_glm",
         "local({ fit <- glm(cbind(ncases, ncontrols) ~ agegp + alcgp, family = binomial, data = esoph, control = glm.control(epsilon = 1e-14, maxit = 100)); fit <- update(fit, start = coef(fit)); list(coefficients = coef(fit), standard_errors = sqrt(diag(vcov(fit))), deviance = deviance(fit), aic = AIC(fit)) })",
         1e-9, 0,
         "Both sides run IRLS to convergence (R with epsilon = 1e-14) and evaluate the covariance at the solution: the R call refits from the converged coefficients because vcov() of a glm uses the working weights of the iteration before the final update (a 1e-8 relative effect on esoph).",
         inputs = "esoph",
         note = "Grouped binomial data: probstat receives the proportions ncases / (ncases + ncontrols) with the totals as prior weights, which is what glm does with a two-column response.")
add_case("regression.glm.poisson", "regression", "probstat.inference.fit_glm",
         "local({ fit <- glm(breaks ~ wool + tension, family = poisson, data = warpbreaks, control = glm.control(epsilon = 1e-14, maxit = 100)); fit <- update(fit, start = coef(fit)); list(coefficients = coef(fit), standard_errors = sqrt(diag(vcov(fit))), deviance = deviance(fit), null_deviance = fit$null.deviance, aic = AIC(fit)) })",
         1e-9, 0,
         "Both sides run IRLS to convergence (R with epsilon = 1e-14) and evaluate the covariance at the solution: the R call refits from the converged coefficients because vcov() of a glm uses the working weights of the iteration before the final update (a 1e-8 relative effect on esoph).",
         inputs = "warpbreaks")
add_case("regression.glm.gamma_log", "regression", "probstat.inference.fit_glm",
         "local({ fit <- glm(Ozone ~ Temp, family = Gamma(link = \"log\"), data = airquality, control = glm.control(epsilon = 1e-14, maxit = 100)); fit <- update(fit, start = coef(fit)); list(coefficients = coef(fit), standard_errors = sqrt(diag(vcov(fit))), deviance = deviance(fit), dispersion = summary(fit)$dispersion, aic = AIC(fit)) })",
         1e-9, 0,
         "Both sides run IRLS to convergence (R with epsilon = 1e-14) and evaluate the covariance at the solution: the R call refits from the converged coefficients because vcov() of a glm uses the working weights of the iteration before the final update (a 1e-8 relative effect on esoph).",
         inputs = "airquality",
         note = "Both use the Pearson dispersion X^2 / (n - p) for standard errors and the deviance / n plug-in for the AIC.")
add_case("regression.glm.gamma_inverse", "regression", "probstat.inference.fit_glm",
         "local({ fit <- glm(Ozone ~ Temp, family = Gamma, data = airquality, control = glm.control(epsilon = 1e-14, maxit = 100)); fit <- update(fit, start = coef(fit)); list(coefficients = coef(fit), standard_errors = sqrt(diag(vcov(fit))), deviance = deviance(fit), aic = AIC(fit)) })",
         1e-9, 0,
         "Both sides run IRLS to convergence (R with epsilon = 1e-14) and evaluate the covariance at the solution: the R call refits from the converged coefficients because vcov() of a glm uses the working weights of the iteration before the final update (a 1e-8 relative effect on esoph).",
         inputs = "airquality")
add_case("regression.glm_nb", "regression", "probstat.inference.fit_negative_binomial_regression",
         "local({ fit <- MASS::glm.nb(breaks ~ wool + tension, data = warpbreaks, control = glm.control(epsilon = 1e-14, maxit = 100)); list(coefficients = coef(fit), theta = fit$theta, log_likelihood = fit$twologlik / 2) })",
         1e-8, 0,
         "glm.nb alternates IRLS with a Newton solve for theta until the changes in likelihood and theta fall below epsilon = 1e-14; probstat runs Newton-Raphson on (beta, log theta) jointly.",
         inputs = "warpbreaks")
add_case("regression.glm_nb.standard_errors", "regression", "probstat.inference.fit_negative_binomial_regression",
         "local({ fit <- MASS::glm.nb(breaks ~ wool + tension, data = warpbreaks, control = glm.control(epsilon = 1e-14, maxit = 100)); th <- MASS::theta.ml(warpbreaks$breaks, fitted(fit), limit = 100, eps = 1e-12); list(standard_errors = sqrt(diag(vcov(fit))), theta_standard_error = attr(th, \"SE\")) })",
         1e-8, 0,
         "Both sides invert the same observed information of theta (and the information of beta given theta) at the converged fit.",
         inputs = "warpbreaks",
         note = "fit$SE.theta is computed inside glm.nb's last theta.ml call with the means of the preceding IRLS fit (6.6e-8 relative from the value at the solution); theta.ml at the final fitted means evaluates it at the solution.")
add_case("regression.rlm_huber", "regression", "probstat.inference.fit_robust_linear_model",
         "local({ fit <- MASS::rlm(stack.loss ~ Air.Flow + Water.Temp + Acid.Conc., data = stackloss, k = 1.345 * 0.6745 / qnorm(0.75), acc = 1e-15, maxit = 500); list(coefficients = coef(fit), scale = fit$s * 0.6745 / qnorm(0.75)) })",
         1e-8, 0,
         "Both sides iterate the same IRLS fixed point from the least-squares start; R stops when the relative residual change is below acc = 1e-15.",
         inputs = "stackloss",
         note = "MASS::rlm scales the residuals by median|r| / 0.6745, probstat by median|r| * 1/qnorm(0.75). Scaling k by 0.6745 / qnorm(0.75) makes the Huber weights identical, so the R call uses that k and converts the reported scale; with MASS's default k = 1.345 the coefficients differ in the sixth digit.")
add_case("regression.nls", "regression", "probstat.optimization.least_squares",
         "local({ fit <- nls(rate ~ SSmicmen(conc, Vm, K), data = Puromycin, control = nls.control(tol = 1e-8, maxiter = 200)); s <- summary(fit); list(estimates = coef(fit), standard_errors = s$coefficients[, 2], sigma = s$sigma) })",
         1e-7, 0,
         "Both sides reach the same least-squares minimum with analytic gradients; R stops when its relative-offset criterion falls below tol = 1e-8, a few 1e-9 relative from the minimum.",
         inputs = "Puromycin",
         note = "probstat: least_squares with the analytic Jacobian and gauss_newton_covariance. SSmicmen supplies the analytic gradient; with the plain formula nls differentiates by forward differences and its standard errors move by 1.7e-8 relative.")
add_case("regression.lme_reml", "regression", "probstat.inference.fit_linear_mixed_model",
         "local({ fit <- nlme::lme(extra ~ group, random = ~ 1 | ID, data = sleep, method = \"REML\", control = nlme::lmeControl(msTol = 1e-14, tolerance = 1e-12, niterEM = 0, msMaxIter = 500)); list(coefficients = nlme::fixef(fit), standard_errors = sqrt(diag(vcov(fit))), residual_variance = fit$sigma^2, intercept_variance = as.numeric(nlme::VarCorr(fit)[1, 1]), log_likelihood = as.numeric(logLik(fit))) })",
         1e-6, 0,
         "Both sides maximize the same restricted likelihood numerically (nlme by nlminb, probstat by BFGS); nlme stops 3e-8 relative from the closed-form REML variances (case regression.lme_reml.closed_form).",
         inputs = "sleep")
add_case("regression.lme_reml.closed_form", "regression", "probstat.inference.fit_linear_mixed_model",
         "local({ d <- sleep$extra[sleep$group == \"2\"] - sleep$extra[sleep$group == \"1\"]; m <- (sleep$extra[sleep$group == \"2\"] + sleep$extra[sleep$group == \"1\"]) / 2; list(residual_variance = var(d) / 2, intercept_variance = (2 * var(m) - var(d) / 2) / 2) })",
         1e-7, 0,
         "In this balanced subjects-by-treatments design the REML variances equal the ANOVA estimators (the residual mean square, and half the excess of the subject mean square over it) when positive; probstat's BFGS on the profiled criterion (tolerance 1e-7) stops a few 1e-9 relative from them.",
         inputs = "sleep")

# ================================================================ survival

add_case("survival.kaplan_meier", "survival", "probstat.inference.survival.kaplan_meier",
         "local({ s <- summary(survfit(Surv(time, status) ~ 1, data = aml, conf.type = \"log-log\")); list(times = s$time, at_risk = s$n.risk, events = s$n.event, survival = s$surv) })",
         1e-12, 0, same_closed_form, inputs = "aml")
add_case("survival.kaplan_meier.interval", "survival", "probstat.inference.survival.kaplan_meier",
         "local({ s <- summary(survfit(Surv(time, status) ~ 1, data = aml, conf.type = \"log-log\")); list(standard_error = s$std.err, lower = s$lower, upper = s$upper) })",
         1e-10, 0, same_closed_form, inputs = "aml",
         note = "Greenwood standard error of S(t) and log-log pointwise interval (conf.type = \"log-log\", probstat's interval; R's default is \"log\").")
add_case("survival.log_rank", "survival", "probstat.inference.survival.log_rank_test",
         "local({ d <- survdiff(Surv(time, status) ~ x, data = aml); list(statistic = d$chisq, observed = d$obs, expected = d$exp, p_value = pchisq(d$chisq, 1, lower.tail = FALSE)) })",
         1e-10, 0, same_closed_form, inputs = "aml")
add_case("survival.cox.efron", "survival", "probstat.inference.survival.fit_cox_model",
         "local({ fit <- coxph(Surv(time, status) ~ x, data = aml, ties = \"efron\", control = coxph.control(eps = 1e-11, iter.max = 100)); list(coefficients = coef(fit), standard_errors = sqrt(diag(vcov(fit))), log_partial_likelihood = fit$loglik[2], null_log_partial_likelihood = fit$loglik[1], score_statistic = fit$score) })",
         1e-9, 0,
         "Both sides run Newton-Raphson on the same Efron partial likelihood to convergence (R with eps = 1e-11).",
         inputs = "aml")
add_case("survival.cox.breslow", "survival", "probstat.inference.survival.fit_cox_model",
         "local({ fit <- coxph(Surv(time, status) ~ x, data = aml, ties = \"breslow\", control = coxph.control(eps = 1e-11, iter.max = 100)); list(coefficients = coef(fit), standard_errors = sqrt(diag(vcov(fit))), log_partial_likelihood = fit$loglik[2], null_log_partial_likelihood = fit$loglik[1], score_statistic = fit$score) })",
         1e-9, 0,
         "Both sides run Newton-Raphson on the same Breslow partial likelihood to convergence (R with eps = 1e-11).",
         inputs = "aml")

# ============================================================== time series

add_case("timeseries.acf", "timeseries", "probstat.timeseries.acf",
         "list(values = acf(lh, lag.max = 10, plot = FALSE)$acf)",
         1e-12, 1e-14, same_closed_form, inputs = "lh")
add_case("timeseries.pacf", "timeseries", "probstat.timeseries.pacf",
         "list(values = pacf(lh, lag.max = 10, plot = FALSE)$acf)",
         1e-10, 1e-13, "Durbin-Levinson recursion on the same autocorrelations.", inputs = "lh")
add_case("timeseries.yule_walker", "timeseries", "probstat.timeseries.yule_walker",
         "list(coefficients = ar.yw(lh, aic = FALSE, order.max = 3)$ar)",
         1e-10, 0, "Durbin-Levinson recursion on the same autocovariances.", inputs = "lh",
         note = "Only the coefficients are compared: ar.yw rescales the innovation variance by n / (n - (p + 1)).")

arima_models <- list(
  ar1 = list(series = "lh", call = "arima(lh, order = c(1, 0, 0), method = \"ML\", optim.control = list(reltol = 1e-15, maxit = 1000))",
             note = "R's \"intercept\" is the process mean, probstat's mean."),
  arma11 = list(series = "LakeHuron", call = "arima(LakeHuron, order = c(1, 0, 1), method = \"ML\", optim.control = list(reltol = 1e-15, maxit = 1000))",
                note = "R's \"intercept\" is the process mean, probstat's mean."),
  ima11 = list(series = "LakeHuron", call = "arima(diff(LakeHuron), order = c(0, 0, 1), include.mean = FALSE, method = \"ML\", optim.control = list(reltol = 1e-15, maxit = 1000))",
               note = "arima(LakeHuron, order = c(0, 1, 1)) approximates the diffuse prior of the differenced state by a variance of kappa = 1e6, which moves ma1 to 0.200254 (1.3e-4 relative); probstat maximizes the exact likelihood of the differenced series, which is what the R call fits.")
)
for (name in names(arima_models)) {
  model <- arima_models[[name]]
  add_case(sprintf("timeseries.arima.%s", name), "timeseries", "probstat.timeseries.fit_arima",
           sprintf("local({ fit <- %s; list(coefficients = coef(fit), sigma2 = fit$sigma2) })", model$call),
           1e-5, 0,
           "Both sides maximize the same exact Gaussian likelihood numerically; R's BFGS on finite-difference gradients stops where the likelihood is flat to about 1e-11, which leaves the coefficients free by about 1e-6 relative.",
           inputs = model$series, note = model$note)
  add_case(sprintf("timeseries.arima.%s.log_likelihood", name), "timeseries",
           "probstat.timeseries.fit_arima",
           sprintf("list(log_likelihood = %s$loglik)", model$call),
           1e-10, 0,
           "The maximized log-likelihood is insensitive to first order to where each optimizer stops.",
           inputs = model$series)
}

# ============================================================ distribution fitting

add_case("fitting.gamma.mle", "fitting", "probstat.inference.fit_distribution",
         "local({ f <- fitdist(airquality$Ozone, \"gamma\", method = \"mle\", control = list(reltol = 1e-15, maxit = 10000)); list(estimates = f$estimate, log_likelihood = f$loglik) })",
         1e-6, 0,
         "fitdistrplus maximizes by Nelder-Mead (reltol 1e-15), which stops 3e-7 relative from the root of the likelihood equations; probstat solves those equations.",
         inputs = "airquality",
         note = "With optim.method = \"BFGS\" (finite-difference gradients) fitdistrplus stops 2.3e-4 relative from the maximum (log-likelihood -541.5376446 against -541.5376424); probstat agrees with the profile root to 1e-15 (case fitting.gamma.mle.profile).")
add_case("fitting.gamma.mle.profile", "fitting", "probstat.inference.fit_distribution",
         "local({ x <- airquality$Ozone; s <- log(mean(x)) - mean(log(x)); k <- uniroot(function(k) log(k) - digamma(k) - s, c(0.1, 10), tol = 1e-15)$root; list(estimates = c(k, k / mean(x))) })",
         1e-12, 0,
         "Both sides solve the profile likelihood equation log(k) - digamma(k) = log(mean x) - mean(log x) to full precision.",
         inputs = "airquality")
add_case("fitting.gamma.mle.standard_errors", "fitting", "probstat.inference.fit_distribution",
         "local({ x <- airquality$Ozone; n <- length(x); s <- log(mean(x)) - mean(log(x)); k <- uniroot(function(k) log(k) - digamma(k) - s, c(0.1, 10), tol = 1e-15)$root; b <- k / mean(x); list(standard_errors = sqrt(diag(solve(n * matrix(c(trigamma(k), -1 / b, -1 / b, k / b^2), 2))))) })",
         1e-6, 0,
         "R evaluates the exact Fisher information of the gamma law at the MLE; probstat inverts a central-difference Hessian of the log-likelihood, accurate to about 2e-7 relative.",
         inputs = "airquality",
         note = "fitdistrplus's standard errors come from optim's Hessian (finite differences with ndeps = 1e-3) and are 1.2e-3 relative from the exact values, so the exact information is the reference.")
add_case("fitting.weibull.mle", "fitting", "probstat.inference.fit_distribution",
         "local({ f <- fitdist(airquality$Ozone, \"weibull\", method = \"mle\", control = list(reltol = 1e-15, maxit = 10000)); list(estimates = f$estimate, log_likelihood = f$loglik) })",
         1e-6, 0,
         "fitdistrplus maximizes by Nelder-Mead (reltol 1e-15), which stops about 4e-8 relative from the root of the likelihood equations; probstat solves those equations.",
         inputs = "airquality")
add_case("fitting.weibull.mle.profile", "fitting", "probstat.inference.fit_distribution",
         "local({ x <- airquality$Ozone; b <- uniroot(function(b) sum(x^b * log(x)) / sum(x^b) - 1 / b - mean(log(x)), c(0.5, 5), tol = 1e-15)$root; list(estimates = c(b, mean(x^b)^(1 / b))) })",
         1e-12, 0,
         "Both sides solve the Weibull profile equation for the shape to full precision; the scale follows in closed form.",
         inputs = "airquality")
add_case("fitting.lnorm.mle", "fitting", "probstat.inference.fit_distribution",
         "local({ f <- fitdist(airquality$Ozone, \"lnorm\", method = \"mle\"); list(estimates = f$estimate, log_likelihood = f$loglik) })",
         1e-12, 0,
         "fitdistrplus and probstat both use the closed-form log-normal MLE.",
         inputs = "airquality")
add_case("fitting.lnorm.mle.standard_errors", "fitting", "probstat.inference.fit_distribution",
         "local({ y <- log(airquality$Ozone); n <- length(y); s <- sqrt(mean((y - mean(y))^2)); list(standard_errors = c(s / sqrt(n), s / sqrt(2 * n))) })",
         1e-6, 0,
         "R evaluates the exact inverse Fisher information; probstat inverts a central-difference Hessian of the log-likelihood, accurate to about 3e-8 relative.",
         inputs = "airquality")
add_case("fitting.gamma.mme", "fitting", "probstat.inference.fit_distribution_moments",
         "list(estimates = fitdist(airquality$Ozone, \"gamma\", method = \"mme\")$estimate)",
         1e-12, 0, same_closed_form, inputs = "airquality",
         note = "fitdistrplus matches the mean and the biased (divisor n) variance, as probstat does.")
add_case("fitting.gev.mle", "fitting", "probstat.risk.fit_gev",
         "local({ g <- ismev::gev.fit(portpirie, show = FALSE, method = \"Nelder-Mead\", maxit = 100000, reltol = 1e-15); list(estimates = g$mle, log_likelihood = -g$nllh) })",
         1e-6, 0,
         "ismev maximizes by Nelder-Mead (reltol 1e-15); the simplex stops on function values, which leaves the parameters free by about 1e-7 relative.",
         inputs = "portpirie",
         note = "ismev and probstat share the sign convention: xi > 0 is the heavy-tailed Frechet type.")

# ================================================================ skipped

add_skip("HoltWinters",
         "probstat's simple and Holt exponential smoothing concentrate the initial level and trend out by least squares, while HoltWinters starts from the first observations (l.start = x[1], or x[2] and x[2] - x[1]); the fitted smoothing parameters and SSE differ by construction.")
add_skip("fisher.test conditional odds ratio",
         "probstat reports the sample odds ratio ad/bc rather than R's conditional maximum-likelihood estimate; the p-values are compared.")
add_skip("mgcv::gam",
         "probstat's additive models use Eilers-Marx P-splines with their own knot placement and identifiability constraint; mgcv's default thin-plate regression splines (and its bs = \"ps\" variant with different knot conventions) do not define the same penalized fit.")
add_skip("lme4::lmer", "lme4 is not installed; nlme::lme is used for the REML comparison.")
add_skip("quantreg::rq", "quantreg is not installed.")
add_skip("car::leveneTest", "car is not installed; the base-R ANOVA of absolute deviations is used instead.")
add_skip("MASS::rlm standard errors",
         "summary.rlm scales sqrt(sum(psi^2) / (n - p)) by Huber's correction factor kappa / mean(psi') and uses (X'WX)^-1; probstat reports the sandwich s^2 n / (n - p) A^-1 B A^-1, so only coefficients and scale are compared.")
add_skip("fitdistrplus standard errors",
         "fitdist inverts optim's finite-difference Hessian (ndeps = 1e-3), 1.2e-3 relative from the exact gamma information on airquality$Ozone; the exact Fisher information evaluated in base R is the reference instead.")
add_skip("ismev::gev.fit standard errors",
         "gev.fit inverts optim's finite-difference Hessian (ndeps = 1e-3); on portpirie its standard errors differ from probstat's central-difference observed information by 4e-6, 1.2e-4, and 2.8e-5 relative (mu, sigma, xi), and base R offers no exact GEV information to arbitrate.")
add_skip("mad() with the default constant",
         "R's default constant 1.4826 is rounded; the case passes constant = 1 / qnorm(0.75), probstat's constant.")

# ================================================================== output

package_versions <- list()
for (p in c("stats", "MASS", "nlme", "survival", "fitdistrplus", "ismev", "e1071")) {
  package_versions[[p]] <- as.character(utils::packageVersion(p))
}
frame_columns <- function(frame) lapply(as.list(frame), function(v) I(if (is.factor(v)) as.character(v) else as.numeric(v)))
datasets_json <- list(
  sleep = frame_columns(sleep), ToothGrowth = frame_columns(ToothGrowth),
  PlantGrowth = frame_columns(PlantGrowth), warpbreaks = frame_columns(warpbreaks),
  mtcars = frame_columns(mtcars), airquality = frame_columns(airquality),
  faithful = frame_columns(faithful), InsectSprays = frame_columns(InsectSprays),
  chickwts = frame_columns(chickwts), stackloss = frame_columns(stackloss),
  esoph = frame_columns(esoph), aml = frame_columns(aml), Puromycin = frame_columns(Puromycin),
  lh = I(lh), LakeHuron = I(LakeHuron), portpirie = I(portpirie),
  depression_x = I(depression_x), depression_y = I(depression_y),
  permeability_x = I(permeability_x), permeability_y = I(permeability_y),
  rounding_times = lapply(seq_len(nrow(rounding_times)), function(i) I(rounding_times[i, ])),
  tea_tasting = lapply(1:2, function(i) I(tea_tasting[i, ])),
  convictions = lapply(1:2, function(i) I(convictions[i, ])),
  job_satisfaction = lapply(1:4, function(i) I(job_satisfaction[i, ])),
  party_by_gender = lapply(1:2, function(i) I(party_by_gender[i, ])),
  ucb_admissions = lapply(1:2, function(i) I(ucb_admissions[i, ])),
  performance = lapply(1:2, function(i) I(performance[i, ])),
  p_values = I(p_values)
)
helpers <- list(
  htest_values = paste(deparse(htest_values), collapse = "\n"),
  abs_deviation_anova = paste(deparse(abs_deviation_anova), collapse = "\n")
)
document <- list(
  generator = "tools/r_oracles/parity.R",
  r_version = R.version.string,
  packages = package_versions,
  comparison = "A value passes when |probstat - R| <= absolute + relative * |R|.",
  case_count = length(cases),
  helpers = helpers,
  datasets = datasets_json,
  cases = cases,
  skipped = skipped
)
dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
connection <- file(output_path, open = "wb")
writeLines(to_json(document), connection, sep = "\n", useBytes = TRUE)
close(connection)
cat(sprintf("wrote %d cases (%d skipped) to %s\n", length(cases), length(skipped), output_path))
