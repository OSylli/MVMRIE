# MVMRIE (Multivariable Mendelian Randomization with Incomplete Measurements on the Exposure Variables)

## Description

This package is used for multivariable Mendelian randomization (MVMR) with  two continuous exposure variables and a continuous outcome, where the exposure variables are potentially unmeasured and subject to detection limits. Specifically, let $Y$ be a continuous outcome, $S_1$ and $S_2$ be two continuous exposure variables that are potentially unmeasured and subject to detection limits, $\mathbf{G}$ be a vector of instrumental variables (IVs) for $S_1$ and $S_2$, and $\mathbf{Z}$ be a vector of measured covariates. Assume that the first component of $\mathbf{Z}$ is 1, and let $\mathbf{X} = (\mathbf{G}^T, \mathbf{Z}^T)^T$. We consider the following models:
$$S_1 = \alpha_1^T \mathbf{X} + \epsilon_1,$$
$$S_2 = \alpha_2^T \mathbf{X} + \epsilon_2,$$
$$Y = \gamma_1 S_1 + \gamma_2 S_2 + \beta^T \mathbf{Z} + \epsilon_Y,$$
where $\alpha_1$, $\alpha_2$, and $\beta$ are regression parameters, $\gamma_1$ and $\gamma_2$ represent the direct causal effect of $S_1$ and $S_2$ on $Y$, respectively, and $(\epsilon_1, \epsilon_2, \epsilon_Y)^T$ is a three-dimensional normal random vector with mean zero and an unstructured covariance matrix. We treat $S_1$ and $S_2$ as potentially missing data. The estimations are carried out using the maximum likelihood estimation method, and the expectation-maximization (EM) algorithm is used to handle the incomplete measurements of the exposures. The estimated covariance matrix is derived from the Louis formula described in Section 8.4 of [Little and Rubin (2019)](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119482260).

## Installation

The `MVMRIE` package requires the following R packages: `Rcpp` (>= 0.11.0), `RcppArmadillo`, `data.table`, `stats`, and `statmod`. You can install them by using the following code:
```{r}
install.packages(c("Rcpp", "RcppArmadillo", "data.table", "stats", "statmod"), dependencies=TRUE)
```

To install `MVMRIE`, you can first install and load the package `devtools` and then run the following code: 
```{r}
install_github("OSylli/MVMRIE")
```
for the latest version. You can also clone the latest repository here and install it yourself using `devtools`.



## Tutorials

The primary function of the package is `MVMRIE_EM`, which has five arguments:

* The first argument `IV.dat` is a required data frame containing data on the IVs for each individual, with a header line indicating the names
of the IVs. Each row corresponds to one individual, and each column
corresponds to one IV. Missing values or `NA`s are not allowed.
* The second argument `pheno.dat` is a required data frame containing data on the outcome and the two exposure variables, with a header line indicating their names. Missing values or `NA`s are not allowed in this data frame. Each row corresponds to one individual, and the individuals should be listed in the same order as in `IV.dat`. This dataset should have five columns. 
  + The first column contains the floating-point values for the continuous outcome. 
  + The second column (integer-valued) and the third column (floating-point valued) contain information on the first exposure variable $S_1$. Specifically, the $i$th element of the second column indicates whether the first exposure of the $i$th subject is observed, beyond detection limits, or unmeasured, and the corresponding value in the third column should be the exact measurement of $S_1$, the detection limit, and a dummy value of -999, respectively. Below is a table showing how the second and the third column should be specified when $S_1$ is observed, beyond detection limits, or unmeasured.
  + The fourth and fifth columns should be specified in the same way for $S_2$.

Measurement type | The 2nd column | The 3rd column
:---------------:|:--------------:|:--------------:|
Measured and detectable | 0 | the exact measurement of $S_1$
Measured but below the lower detection limit | 1 | the lower detection limit	|
Measured but above the upper detection limit | 2 | the upper detection limit	|
Unmeasured | 3 | -999 

* The third argument `covar.dat` is an optional data frame containing data on the measured
covariates, with a header line indicating the names of these variables. Each row corresponds to one individual, and the individuals should be listed in the same order as in `IV.dat`. Each column corresponds to one covariate. Missing values or `NA`s are not allowed. The default value is `NULL`.
* The fourth argument `epsilon` is a positive floating-point value indicating the convergence threshold of the EM algorithm. The iterations will be terminated if the Euclidean distance between the parameter values at two successive iterations is less than this value. The default value is `1e-5`.
* The last argument `num.ghq` is a positive integer indicating the number of quadrature nodes to be used when approximating the integrals with Gauss-Hermite quadrature during the computation.

### Output values

When the function `MVMRIE_EM` is run correctly, a list with the following components will be returned:

* `results_reg` contains the point estimates for $\alpha_1, \alpha_2, \beta, \gamma_1$, and
$\gamma_2$. Standard error estimates and $p$-values are also provided. The inference on the direct causal effects of interest is presented in the last two rows.
* `results_var` contains the estimates for the variance components (i.e., the variances of $\epsilon_1, \epsilon_2$, and $\epsilon_Y$ and their pairwise correlation).
* `full_cov_mat` is the entire estimated covariance matrix derived from the Louis formula.

### Example

We show the implementation of the proposed function with the following example. Firstly, we load the example dataset.
```{r}
# set the working directory as the path where the example data were stored
library(MVMRIE)
setwd(system.file("extdata", package="MVMRIE"))

data <- read.table("example_data.txt", header = TRUE, sep = "\t")
dim(data)
head(data)
tail(data)
unique(data$type1)
unique(data$type2)
```

The example dataset has 11 columns:

* The first column contains the measurements of the continuous outcome of interest.
* The second column indicates the measurement type of the first exposure $S_1$, where the value `0` suggests that $S_1$ is measured and detectable, the value `1` means that $S_1$ is measured but below the lower detection limit, and the value `3` means that $S_1$ is unmeasured.
* The third column shows the exact measurements of $S_1$ if it is measured and detectable, the lower detection limit if $S_1$ is measured but below the lower detection limit, and a dummy value of `-999` if it is unmeasured.
* The fourth and fifth columns are specified for $S_2$ under the same rule as the second and third columns for $S_1$.
* The sixth, seventh, and eighth columns contain data on the three IVs that are used.
* The last three columns contain measurements on the measured covariates of age, gender, and the first principal component for ancestry.

Then, we can prepare for the function arguments:
```{r}
# The argument `IV.dat` is a data frame containing measurements on the IVs
IV.dat <- data[, c("IV1", "IV2", "IV3")]

# The argument `pheno.dat` is a data frame containing information on the outcome and the exposure variables
pheno.dat <- data[, c("outcome", "type1", "exposure1", "type2", "exposure2")]

# The argument `covar.dat` is a data frame containing measurements on the measured covariates
covar.dat <- data[, c("AGE", "GENDER", "PC")]

# Analyze the example data with the function `MVMRIE_EM` in the package, using 50 nodes for Gauss-Hermite quadrature 
fit <- MVMRIE_EM(IV.dat, pheno.dat, covar.dat, num.ghq = 50)
```

We can then check the results as following:
```{r}
# The inference on the regression parameters
fit$results_reg

# The estimates of the variance components
fit$results_var

# The full estimated covariance matrix derived from the Louis formula
fit$full_cov_mat
```

Below is the output (of `results_reg` and `results_var`) for the analysis using the example data. 

```
> fit$results_reg
                          Estimate Std. Error      p-value
IV1 on exposure1        0.23944531 0.02531409 0.000000e+00
IV2 on exposure1        0.21648410 0.02335372 0.000000e+00
IV3 on exposure1       -0.01410285 0.02253647 5.314600e-01
intercept on exposure1  0.24585141 0.04852811 4.059103e-07
AGE on exposure1        0.01438129 0.05566878 7.961470e-01
GENDER on exposure1     0.14459328 0.03206136 6.486058e-06
PC on exposure1         0.16634430 0.01626621 0.000000e+00
IV1 on exposure2       -0.00844521 0.02481795 7.336409e-01
IV2 on exposure2        0.15915183 0.02309518 5.535128e-12
IV3 on exposure2        0.28073924 0.02254961 0.000000e+00
intercept on exposure2  0.15047078 0.04822417 1.807062e-03
AGE on exposure2        0.19188447 0.05489299 4.729803e-04
GENDER on exposure2     0.16758285 0.03163900 1.179029e-07
PC on exposure2         0.14742197 0.01598611 0.000000e+00
intercept on outcome    0.13612848 0.04294992 1.527171e-03
AGE on outcome          0.19232274 0.03988694 1.423420e-06
GENDER on outcome       0.14755744 0.02446544 1.626844e-09
PC on outcome           0.15779060 0.01529310 0.000000e+00
exposure1 on outcome    0.12356366 0.05508110 2.487720e-02
exposure2 on outcome    0.18169084 0.05184500 4.574643e-04

> fit$results_var
                                       Estimate
variance of error term (exposure 1)  1.02609734
variance of error term (exposure 2)  1.03640711
variance of error term (outcome)     1.05710577
correlation (exposure 1, exposure 2) 0.03237070
correlation (exposure 1, outcome)    0.03334973
correlation (exposure 2, outcome)    0.05542800
```



## Support

If you have any further questions or problems with installing or running `MVMRIE`, please email me at <yilunli1997@gmail.com>.
