test_that("NA values in the input dataset", {
  # model construction
  # S1 = 0.2 G1 + 0.15 G2 +        + 0.15 + e1
  # S2 =        + 0.15 G2 + 0.2 G3 + 0.15 + e2
  #  Y = 0.15 S1 + 0.15 S2 + 0.15 + eY
  n <- 9000
  prob1 <- 0.3
  prob2 <- 0.4
  prob3 <- 0.5

  # independent variables and error terms
  e1 <- rnorm(n)
  e2 <- rnorm(n)
  eY <- rnorm(n)
  G1 <- rbinom(n, 2, prob1)
  G2 <- rbinom(n, 2, prob2)
  G3 <- rbinom(n, 2, prob3)

  # parameters
  alpha1 <- 0.2
  alpha2 <- 0.15
  alpha3 <- 0.2
  gamma1 <- 0.15
  gamma2 <- 0.15
  intercept <- 0.15

  # dependent variables
  S1 <- alpha1 * G1 + alpha2 * G2 + intercept + e1
  S2 <- alpha3 * G3 + alpha2 * G2 + intercept + e2
  R1 <- rep(0, n)
  R2 <- rep(0, n)
  Y <- gamma1 * S1 + gamma2 * S2 + intercept + eY
  Y[1 : (0.01*n)] <- NA

  IV.dat <- data.frame(IV1 = G1, IV2 = G2, IV3 = G3)
  pheno.dat <- data.frame(outcome = Y, type1 = R1, exposure1 = S1, type2 = R2, exposure2 = S2)
  expect_error(MVMRIE_EM(IV.dat, pheno.dat, num.ghq = 50), "NA is not allowed in the phenotype dataset.")
})


test_that("different number of observations", {
  # model construction
  # S1 = 0.2 G1 + 0.15 G2 +        + 0.15 + e1
  # S2 =        + 0.15 G2 + 0.2 G3 + 0.15 + e2
  #  Y = 0.15 S1 + 0.15 S2 + 0.15 + eY
  n <- 9000
  prob1 <- 0.3
  prob2 <- 0.4
  prob3 <- 0.5

  # independent variables and error terms
  e1 <- rnorm(n)
  e2 <- rnorm(n)
  eY <- rnorm(n)
  G1 <- rbinom(n, 2, prob1)
  G2 <- rbinom(n, 2, prob2)
  G3 <- rbinom(n, 2, prob3)

  # parameters
  alpha1 <- 0.2
  alpha2 <- 0.15
  alpha3 <- 0.2
  gamma1 <- 0.15
  gamma2 <- 0.15
  intercept <- 0.15

  # dependent variables
  S1 <- alpha1 * G1 + alpha2 * G2 + intercept + e1
  S2 <- alpha3 * G3 + alpha2 * G2 + intercept + e2
  R1 <- rep(0, n)
  R2 <- rep(0, n)
  Y <- gamma1 * S1 + gamma2 * S2 + intercept + eY

  IV.dat <- data.frame(IV1 = G1, IV2 = G2, IV3 = G3)
  IV.dat <- IV.dat[1:(0.5*n), ]
  pheno.dat <- data.frame(outcome = Y, type1 = R1, exposure1 = S1, type2 = R2, exposure2 = S2)
  expect_error(MVMRIE_EM(IV.dat, pheno.dat, num.ghq = 50), "The instrumental variable dataset and the phenotype dataset have different numbers of observations.")
})


test_that("Wrong specifications of the measurement type", {
  # model construction
  # S1 = 0.2 G1 + 0.15 G2 +        + 0.15 + e1
  # S2 =        + 0.15 G2 + 0.2 G3 + 0.15 + e2
  #  Y = 0.15 S1 + 0.15 S2 + 0.15 + eY
  n <- 9000
  prob1 <- 0.3
  prob2 <- 0.4
  prob3 <- 0.5

  # independent variables and error terms
  e1 <- rnorm(n)
  e2 <- rnorm(n)
  eY <- rnorm(n)
  G1 <- rbinom(n, 2, prob1)
  G2 <- rbinom(n, 2, prob2)
  G3 <- rbinom(n, 2, prob3)

  # parameters
  alpha1 <- 0.2
  alpha2 <- 0.15
  alpha3 <- 0.2
  gamma1 <- 0.15
  gamma2 <- 0.15
  intercept <- 0.15

  # dependent variables
  S1 <- alpha1 * G1 + alpha2 * G2 + intercept + e1
  S2 <- alpha3 * G3 + alpha2 * G2 + intercept + e2
  R1 <- rep(0, n)
  R2 <- rep(0, n)
  Y <- gamma1 * S1 + gamma2 * S2 + intercept + eY

  S1[1 : (0.5 * n)] <- -999
  R1[1 : (0.5 * n)] <- 4

  detection_limit <- 0
  for(i in ((0.5 * n + 1):n)){
    if(S1[i] < detection_limit){
      S1[i] <- detection_limit
      R1[i] <- 1
    }
  }

  IV.dat <- data.frame(IV1 = G1, IV2 = G2, IV3 = G3)
  pheno.dat <- data.frame(outcome = Y, type1 = R1, exposure1 = S1, type2 = R2, exposure2 = S2)
  expect_error(MVMRIE_EM(IV.dat, pheno.dat, num.ghq = 50), "Values in the 2nd column of the phenotype dataset should only be 0, 1, 2, or 3.")
})


