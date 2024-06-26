# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

make_initial2 <- function(Y, S1, S2, Z, X) {
    .Call('_MVMRIE_make_initial2', PACKAGE = 'MVMRIE', Y, S1, S2, Z, X)
}

MVMR_estimation_EM <- function(Y, S1, S2, R1, R2, Z, X, alpha1_init, alpha2_init, beta_init, gamma1_init, gamma2_init, Sigma_init, epsilon, QuaWeight, QuaPoint) {
    .Call('_MVMRIE_MVMR_estimation_EM', PACKAGE = 'MVMRIE', Y, S1, S2, R1, R2, Z, X, alpha1_init, alpha2_init, beta_init, gamma1_init, gamma2_init, Sigma_init, epsilon, QuaWeight, QuaPoint)
}

