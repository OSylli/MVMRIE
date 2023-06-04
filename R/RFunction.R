#' Multivariable Mendelian Randomization with Incomplete Measurements on the Exposure Variables
#'
#' This function is used for multivariable Mendelian randomization (MVMR) with
#' two continuous exposure variables and a continuous outcome, where the exposure
#' variables are potentially unmeasured and subject to detection limits.
#'
#' @param IV.dat A required data frame containing data on the instrumental
#' variables (IVs) for each individual, with a header line indicating the names
#' of the IVs. Each row corresponds to one individual, and each column
#' corresponds to one IV. Missing values or \code{NA}s are not allowed.
#' @param pheno.dat A required data frame containing data on the outcome and the
#' two exposure variables, with a header line indicating their names. Missing
#' values or \code{NA}s are not allowed in this data frame. Each row corresponds to one individual,
#' and the individuals should be listed in the same order as in \code{IV.dat}.
#'
#' The dataset \code{pheno.dat} should have five columns. The 1st column
#' contains the floating-point values for the continuous outcome. The 2nd column
#' (integer-valued) and the 3rd column (floating-point valued) contain
#' information on the first exposure variable, and the 4th column (integer-valued)
#' and the 5th column (floating-point valued) should be similarly
#' specified for the other exposure. See the "Details" section for instructions
#' on specifying these columns.
#' @param covar.dat An optional data frame containing data on the measured
#' covariates, with a header line indicating the names of these variables.
#' Each row corresponds to one individual, and the individuals should be listed
#' in the same order as in \code{IV.dat}. Each column corresponds to one
#' covariate. Missing values or \code{NA}s are not allowed. The default value is \code{NULL}.
#' @param epsilon A positive floating-point value indicating the convergence
#' threshold of the expectation-maximization (EM) algorithm. The iterations will be terminated if the
#' Euclidean distance between the parameter values at two successive iterations
#' is less than this value. The default value is \code{1e-5}.
#' @param num.ghq A positive integer indicating the number of quadrature nodes
#' to be used when approximating the integrals with Gauss-Hermite quadrature
#' during the computation.
#'
#' @details
#' \loadmathjax
#' \strong{1) Model and Method}
#'
#' Let \mjeqn{Y}{ASCII representation} be a continuous outcome,
#' \mjeqn{S_1}{ASCII representation} and \mjeqn{S_2}{ASCII representation} be
#' two continuous exposure variables that are potentially unmeasured and subject
#' to detection limits, \mjeqn{\mathbf{G}}{ASCII representation} be a vector of
#' IVs for \mjeqn{S_1}{ASCII representation} and \mjeqn{S_2}{ASCII representation},
#' and \mjeqn{\mathbf{Z}}{ASCII representation} be a vector of measured
#' covariates. Assume that the first component of \mjeqn{\mathbf{Z}}{ASCII representation}
#' is 1, and let \mjeqn{\mathbf{X} = (\mathbf{G}^T, \mathbf{Z}^T)^T}{ASCII representation}.
#' Consider the following models:
#' \mjdeqn{S_1 = \alpha_1^T \mathbf{X} + \epsilon_1,}{ASCII representation}
#' \mjdeqn{S_2 = \alpha_2^T \mathbf{X} + \epsilon_2,}{ASCII representation}
#' \mjdeqn{Y = \gamma_1 S_1 + \gamma_2 S_2 + \beta^T \mathbf{Z} + \epsilon_Y,}{ASCII representation}
#' where \mjeqn{\alpha_1}{ASCII representation}, \mjeqn{\alpha_2}{ASCII representation},
#' and \mjeqn{\beta}{ASCII representation} are regression parameters,
#' \mjeqn{\gamma_1}{ASCII representation} and \mjeqn{\gamma_2}{ASCII representation}
#' represent the direct causal effect of \mjeqn{S_1}{ASCII representation} and
#' \mjeqn{S_2}{ASCII representation} on \mjeqn{Y}{ASCII representation},
#' respectively, and \mjeqn{(\epsilon_1, \epsilon_2, \epsilon_Y)^T}{ASCII representation}
#' is a three-dimensional normal random vector with mean zero and an
#' unstructured covariance matrix. We treat \mjeqn{S_1}{ASCII representation}
#' and \mjeqn{S_2}{ASCII representation} as potentially missing data. The
#' estimations are carried out using the maximum likelihood estimation method,
#' and the EM algorithm is used to handle the incomplete measurements on the exposures.
#'
#' \strong{2) The \code{pheno.dat} argument}
#'
#' For the argument \code{pheno.dat}, the \mjeqn{i}{ASCII representation}th
#' element of the 2nd column indicates whether the first exposure
#' \mjeqn{S_1}{ASCII representation} of the \mjeqn{i}{ASCII representation}th
#' subject is observed, beyond detection limits, or unmeasured, and the
#' corresponding value in the 3rd column should be the exact measurement of
#' \mjeqn{S_1}{ASCII representation}, the detection limit, and a dummy value of
#' -999, respectively. Below is a table showing how the 2nd and the 3rd column
#' should be specified when \mjeqn{S_1}{ASCII representation} is observed,
#' beyond detection limits, or unmeasured.
#'
#' |						                                   | The 2nd column | The 3rd column		        |
#' |:--------------------------------------------- |:--------------:|:-------------------------:|
#' | Measured and detectable			                 |       0        | the exact measurement	    |
#' | Measured but below the lower detection limit	 |       1        | the lower detection limit	|
#' | Measured but above the upper detection limit	 |       2        | the upper detection limit	|
#' | Unmeasured					                           |       3        | -999				              |
#'
#' The 4th and 5th columns of \code{pheno.dat} should be specified in
#' the same way for \mjeqn{S_2}{ASCII representation}.
#'
#'
#' @return
#' A list with the following components will be returned:
#' \itemize{
#' \item{\code{results_reg} contains the point estimates for
#' \mjeqn{\alpha_1, \alpha_2, \beta, \gamma_1}{ASCII representation}, and
#' \mjeqn{\gamma_2}{ASCII representation}. Standard error estimates and p-values
#' are also provided. The inference on the direct causal effects of interest is
#' presented in the last two rows.}
#' \item{\code{results_var} contains the estimates for the variance components
#' (i.e., the variances of \mjeqn{\epsilon_1, \epsilon_2}{ASCII representation},
#' and \mjeqn{\epsilon_Y}{ASCII representation} and their pairwise correlation).}
#' \item{\code{full_cov_mat} is the entire estimated covariance matrix
#' derived from the Louis formula (Little & Rubin, 2019).}
#' }
#'
#' @importFrom stats pchisq
#' @importFrom statmod gauss.quad
#'
#' @references
#' Little, R. J., & Rubin, D. B. (2019). Statistical analysis with missing data (3rd Edition). John Wiley & Sons.
#'
#' @useDynLib MVMRIE
#' @export
MVMRIE_EM <- function(IV.dat, pheno.dat, covar.dat = NULL, epsilon = 1e-5, num.ghq){
  # Gauss-Hermite Quadrature weights and nodes
  GHQ <- gauss.quad(num.ghq, kind = "hermite")
  GHQ_weight <- GHQ$weights
  GHQ_node <- GHQ$nodes

  ## input column names
  outcome.name <- colnames(pheno.dat)[1]
  exposure1.name <- colnames(pheno.dat)[3]
  exposure2.name <- colnames(pheno.dat)[5]

  ## checking the input datasets - # observations
  n <- nrow(IV.dat)
  if(nrow(pheno.dat) != n){
    stop("The instrumental variable dataset and the phenotype dataset have different numbers of observations.")
  }

  ## checking the input datasets - NA's
  if(sum(is.na(IV.dat)) > 0){
    stop("NA is not allowed in the instrumental variable dataset.")
  }
  if(sum(is.na(pheno.dat)) > 0){
    stop("NA is not allowed in the phenotype dataset.")
  }
  if(!is.null(covar.dat)){
    if(sum(is.na(covar.dat)) > 0){
      stop("NA is not allowed in the covariate variables dataset.")
    }
  }

  ## Preparations for the arguments of the CPP function
  #### Outcome (Y)
  Y <- as.vector(pheno.dat[, 1])

  #### Exposure (S)
  S1 <- as.vector(pheno.dat[, 3])
  R1 <- as.vector(pheno.dat[, 2])
  Check_R1 <- (R1 != 0) & (R1 != 1) & (R1 != 2) & (R1 != 3)
  if(sum(Check_R1) > 0){
    stop("Values in the 2nd column of the phenotype dataset should only be 0, 1, 2, or 3.")
  }

  S2 <- as.vector(pheno.dat[, 5])
  R2 <- as.vector(pheno.dat[, 4])
  Check_R2 <- (R2 != 0) & (R2 != 1) & (R2 != 2) & (R2 != 3)
  if(sum(Check_R2) > 0){
    stop("Values in the 4th column of the phenotype dataset should only be 0, 1, 2, or 3.")
  }

  #### intercept & the independent variables for the model equations
  Z <- data.frame(intercept = rep(1, n))
  if(!is.null(covar.dat)){
    if(nrow(covar.dat) != n){
      stop("The instrumental variable dataset and the covariate dataset have different numbers of observations.")
    }
    Z <- cbind(Z, covar.dat)
  }
  X <- cbind(IV.dat, Z)

  colnames_Z <- colnames(Z)
  colnames_X <- colnames(X)
  Z <- as.matrix(Z)
  X <- as.matrix(X)

  ## Preparation for Imp-Mid
  R1_sub <- R1[which(R1 != 3 & R2 != 3)]
  R2_sub <- R2[which(R1 != 3 & R2 != 3)]
  Y_sub <- Y[which(R1 != 3 & R2 != 3)]
  S1_sub <- S1[which(R1 != 3 & R2 != 3)]
  S1_sub[which(R1_sub == 1)] <- S1_sub[which(R1_sub == 1)] - log(2)
  S2_sub <- S2[which(R1 != 3 & R2 != 3)]
  S2_sub[which(R2_sub == 1)] <- S2_sub[which(R2_sub == 1)] - log(2)
  Z_sub <- Z[which(R1 != 3 & R2 != 3), ]
  X_sub <- X[which(R1 != 3 & R2 != 3), ]

  Z_sub <- as.matrix(Z_sub)
  X_sub <- as.matrix(X_sub)

  ## Analysis - Imputation at mid-point
  fit0 <- make_initial2(Y_sub, S1_sub, S2_sub, Z_sub, X_sub)

  ## Analysis - Proposed method using the EM algorithm
  gamma1_index <- ncol(X_sub) + ncol(X_sub) + ncol(Z_sub) + 1
  EM_fit <- MVMR_estimation_EM(Y, S1, S2, R1, R2, Z, X, fit0[[1]], fit0[[2]], fit0[[3]], fit0[[4]], fit0[[5]], fit0[[6]], epsilon, GHQ_weight, GHQ_node)

  ## output row & column names
  output.colnames <- c("Estimate", "Std. Error", "p-value")
  rowname1 <- paste(colnames_X, " on ", exposure1.name, sep = "", collapse = NULL)
  rowname2 <- paste(colnames_X, " on ", exposure2.name, sep = "", collapse = NULL)
  rowname3 <- paste(colnames_Z, " on ", outcome.name, sep = "", collapse = NULL)
  rowname4 <- paste(exposure1.name, " on ", outcome.name, sep = "", collapse = NULL)
  rowname5 <- paste(exposure2.name, " on ", outcome.name, sep = "", collapse = NULL)
  output.rownames <- c(rowname1, rowname2, rowname3, rowname4, rowname5)

  ## result data.frame & List
  results_reg <- matrix(NA, nrow = length(output.rownames), ncol = length(output.colnames), dimnames = list(output.rownames, output.colnames))
  results_var <- matrix(NA, nrow = 6, ncol = 1, dimnames = list(c("variance of error term (exposure 1)", "variance of error term (exposure 2)",
                                                                  "variance of error term (outcome)", "correlation (exposure 1, exposure 2)",
                                                                  "correlation (exposure 1, outcome)", "correlation (exposure 2, outcome)"), c("Estimate")))

  #### estimates
  results_reg[, 1] <- c(EM_fit[[1]], EM_fit[[2]], EM_fit[[3]], EM_fit[[4]], EM_fit[[5]])

  #### standard error estimates
  results_reg[, 2] <- sqrt(diag(EM_fit[[7]])[1 : length(output.rownames)])

  #### two-sided p-value
  z_score <- results_reg[, 1] / results_reg[, 2]
  results_reg[ ,3] <- 1 - pchisq(z_score^2, 1)

  #### variance components
  results_var[1, 1] <- EM_fit[[6]][1, 1] # var-S1
  results_var[2, 1] <- EM_fit[[6]][2, 2] # var-S2
  results_var[3, 1] <- EM_fit[[6]][3, 3] # var-Y
  results_var[4, 1] <- EM_fit[[6]][1, 2] # corr-12
  results_var[5, 1] <- EM_fit[[6]][1, 3] # corr-1Y
  results_var[6, 1] <- EM_fit[[6]][2, 3] # corr-2Y

  #### full covariance matrix
  full_para_name <- c(output.rownames, rownames(results_var))
  full_cov_mat <- EM_fit[[7]]
  dimnames(full_cov_mat) <- list(full_para_name, full_para_name)

  results_list <- list(results_reg = results_reg, results_var = results_var, full_cov_mat = full_cov_mat)
  return(results_list)
}

