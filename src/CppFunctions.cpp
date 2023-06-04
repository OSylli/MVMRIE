#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <iostream>

using namespace std;
using namespace Rcpp;


// [[Rcpp::export]]
List make_initial2(arma::vec Y, arma::vec S1, arma::vec S2, arma::mat Z, arma::mat X){
  int n = Y.n_elem;
  int k = Z.n_cols + 2;
  
  //cout << "1" << "  ";
  // Stage-1 Regression
  arma::vec alpha1_init = (X.t() * X).i() * X.t() * S1;
  arma::vec alpha2_init = (X.t() * X).i() * X.t() * S2;
  
  //cout << "2" << "  ";
  // Fitted Values of Stage-1 Regression
  arma::mat PX = X * (X.t() * X).i() * X.t();
  arma::vec S1hat = PX * S1;
  arma::vec S2hat = PX * S2;
  arma::mat W = arma::join_rows(arma::join_rows(S1hat, S2hat), Z);
  
  //cout << "3" << "  ";
  // Stage-2 Regression
  arma::vec betagamma = (W.t() *  W).i() * W.t() * Y;
  arma::vec beta_init = betagamma.subvec(2, betagamma.n_elem - 1);
  double gamma1_init = betagamma(0);
  double gamma2_init = betagamma(1);
  
  //cout << "4" << "  ";
  // Initial guess of the variance components
  arma::vec resS1 = S1 - X * alpha1_init;
  arma::vec resS2 = S2 - X * alpha2_init;
  arma::vec resY = Y - gamma1_init * S1 - gamma2_init * S2 - Z * beta_init;
  
  //cout << "5" << "  ";
  double Sigma11 = arma::as_scalar(resS1.t() * resS1) / (n-k);
  double Sigma12 = arma::as_scalar(resS1.t() * resS2) / (n-k);
  double Sigma22 = arma::as_scalar(resS2.t() * resS2) / (n-k);
  double Sigma13 = arma::as_scalar(resS1.t() * resY) / (n-k);
  double Sigma23 = arma::as_scalar(resS2.t() * resY) / (n-k);
  double Sigma33 = arma::as_scalar(resY.t() * resY) / (n-k);
  arma::mat Sigma_init = { {Sigma11, Sigma12, Sigma13}, {Sigma12, Sigma22, Sigma23}, {Sigma13, Sigma23, Sigma33} };
  
  // SEE of the TSLS estimators of \beta and gamma
  arma::mat X_TSLS = arma::join_rows(arma::join_rows(S1, S2), Z);
  arma::mat PZ_TSLS = X * (X.t() * X).i() * X.t();
  arma::mat TSLS_cov = Sigma33 * (X_TSLS.t() * PZ_TSLS * X_TSLS).i();
  
  return List::create(alpha1_init, alpha2_init, beta_init, gamma1_init, gamma2_init, Sigma_init, TSLS_cov);
}



// Find the index that satisfies s[index - 1] <= val && s[index] > val for vector s in an increasing order 
int find_largest_index(arma::vec point_s, double val){
  int K = point_s.n_elem;
  int indexl = 0;
  int indexr = K - 1;
  int index = floor((indexl + indexr) / 2);
  while((point_s[index - 1] > val) || (point_s[index] <= val)){
    if(point_s[index - 1] > val){
      indexr = index;
      index = floor((indexl + indexr) / 2);
    }
    else{
      indexl = index;
      index = floor((indexl + indexr) / 2);
    }
  }
  return index;
}



// Find the index that satisfies s[index] >= val && s[index - 1] < val for vector s in an increasing order
int find_smallest_index(arma::vec point_s, double val){
  int K = point_s.n_elem;
  int indexl = 0;
  int indexr = K - 1;
  int index = floor((indexl + indexr) / 2);
  while((point_s[index - 1] >= val) || (point_s[index] < val)){
    if(point_s[index - 1] >= val){
      indexr = index;
      index = floor((indexl + indexr) / 2);
    }
    else{
      indexl = index;
      index = floor((indexl + indexr) / 2);
    }
  }
  return index;
}



// E-step
arma::mat CondExpect_update(arma::vec alpha1, arma::vec alpha2, arma::vec beta, double gamma1, double gamma2, arma::mat Sigma,
                            arma::vec Y, arma::vec S1, arma::vec S2, arma::ivec R1, arma::ivec R2, arma::mat Z, arma::mat X, 
                            arma::vec weight_w, arma::vec point_s){
  // 1) Relationships between Rj and Sj (j = 1,2):
  //// if Rji = 0: Sji = <observed measurement>
  //// if Rji = 1: Sji = <lower detection limit>
  //// if Rji = 2: Sji = <upper detection limit>
  //// if Rji = 3: Sji = <-999>
  
  // 2) Matrix Ehat: storing the conditional expectations results:
  // The five columns correspond to E(S_1i), E(S_2i), E(S_1i^2), E(S_1iS_2i) and E(S_2i^2), respectively.
  int n = Y.n_elem;
  arma::mat Ehat = arma::zeros(n, 5);
  
  // 3) Notations and Definitions:
  //// Inverse of Sigma and its blocks
  arma::mat Sigma_inv = Sigma.i();
  arma::mat Sigma_inv11 = Sigma_inv.submat(0, 0, 1, 1);
  arma::mat Sigma_inv12 = Sigma_inv.submat(0, 2, 1, 2);
  arma::mat Sigma_inv21 = Sigma_inv12.t();
  double Sigma_inv22 = Sigma_inv(2, 2);
  //// Vector gamma
  arma::vec gamma = arma::zeros(2);
  gamma(0) = gamma1;
  gamma(1) = gamma2;
  //// Matrix alphaX_temp
  arma::mat alphaX_temp = arma::join_cols(alpha1.t() * X.t(), alpha2.t() * X.t());
  //// Matrix A
  arma::mat A = arma::eye(2, 2);
  double a11, a12, a22, a1giv2, a2giv1;
  //// Matrix B, the i-th column of which is equal to bi
  arma::mat B = arma::zeros(2, n);
  //// Matrix MU, the i-th column of which is equal to mu_i
  arma::mat MU = arma::zeros(2, n);
  //// Other quantities:
  ////// variables needed for computations in different scenarios
  double mu2giv1i, mu1giv2i, L2giv1i, L1giv2i, L20i, L10i, U2giv1i, U1giv2i, U20i, U10i;
  ////// number of quadrature points
  int K = point_s.n_elem;
  ////// index for summation
  int index;
  ////// other variables needed when both exposure variables are beyond detection limits
  double I0, I1, I2, I3, I4, I5;
  arma::vec xi1i, xi2i, tilde_mu1giv2i;
  arma::mat vec;
  //cout << "3-1-1" << endl;
  
  // 4) Update Matrix A
  A = (Sigma_inv11 - Sigma_inv12 * gamma.t() - gamma * Sigma_inv21 + gamma * Sigma_inv22 * gamma.t()).i();
  a11 = A(0, 0);
  a12 = A(0, 1);
  a22 = A(1, 1);
  a1giv2 = a11 - a12 * a12 / a22;
  a2giv1 = a22 - a12 * a12 / a11;
  //cout << "3-1-2" << endl;
  
  // 5) Update Matrix B, Matrix MU and Vector bi, Vector Mu_i
  B = (Sigma_inv11 - gamma * Sigma_inv21) * alphaX_temp - (Sigma_inv12 - gamma * Sigma_inv22) * (Y.t() - beta.t() * Z.t());
  MU = A * B;
  //cout << "3-1-3" << endl;
  
  // 6) Case-by-case calculations:
  //// Attention: quadrature points should be sorted in an increasing order!
  for(int i=0; i<n; i++){
    if(R1(i) == 0){
      if(R2(i) == 0){
        vec = {S1(i), S2(i), S1(i)*S1(i), S1(i)*S2(i), S2(i)*S2(i)};
        Ehat.row(i) = vec;
      }
      else if(R2(i) == 3){
        mu2giv1i = MU(1, i) + (S1(i) - MU(0, i)) * a12 / a11;
        vec = {S1(i), mu2giv1i, S1(i)*S1(i), S1(i)*mu2giv1i, mu2giv1i*mu2giv1i + a2giv1};
        Ehat.row(i) = vec;
      }
      else if(R2(i) == 1){
        mu2giv1i = MU(1, i) + (S1(i) - MU(0, i)) * a12 / a11;
        L2giv1i = (S2(i) - mu2giv1i) / sqrt(a2giv1);
        vec = {S1(i), mu2giv1i - sqrt(a2giv1) * arma::normpdf(L2giv1i)/arma::normcdf(L2giv1i), S1(i)*S1(i),
               S1(i) * (mu2giv1i - sqrt(a2giv1) * arma::normpdf(L2giv1i)/arma::normcdf(L2giv1i)),
               mu2giv1i * mu2giv1i + a2giv1 - (2*mu2giv1i*sqrt(a2giv1) + a2giv1*L2giv1i) * arma::normpdf(L2giv1i)/arma::normcdf(L2giv1i)};
        Ehat.row(i) = vec;
      }
      else if(R2(i) == 2){
        mu2giv1i = MU(1, i) + (S1(i) - MU(0, i)) * a12 / a11;
        U2giv1i = (-S2(i) + mu2giv1i) / sqrt(a2giv1);
        vec = {S1(i), mu2giv1i + sqrt(a2giv1) * arma::normpdf(U2giv1i)/arma::normcdf(U2giv1i), S1(i)*S1(i),
               S1(i) * (mu2giv1i + sqrt(a2giv1) * arma::normpdf(U2giv1i)/arma::normcdf(U2giv1i)),
               mu2giv1i * mu2giv1i + a2giv1 + (2*mu2giv1i*sqrt(a2giv1) - a2giv1*U2giv1i) * arma::normpdf(U2giv1i)/arma::normcdf(U2giv1i)};
        Ehat.row(i) = vec;
      }
      else{
        stop("Inadmissible values: elements of vector R2 should be chosen from (0, 1, 2, 3).");
      }
    }
    else if(R1(i) == 3){
      if(R2(i) == 0){
        mu1giv2i = MU(0, i) + (S2(i) - MU(1, i)) * a12 / a22;
        vec = {mu1giv2i, S2(i), mu1giv2i*mu1giv2i + a1giv2, S2(i)*mu1giv2i, S2(i)*S2(i)};
        Ehat.row(i) = vec;
      }
      else if(R2(i) == 3){
        vec = {MU(0, i), MU(1, i), MU(0, i)*MU(0, i) + a11, MU(0, i)*MU(1, i) + a12, MU(1, i)*MU(1, i) + a22};
        Ehat.row(i) = vec;
      }
      else if(R2(i) == 1){
        L20i = (S2(i) - MU(1, i)) / sqrt(a22);
        vec = {MU(0, i) - a12 / sqrt(a22) * arma::normpdf(L20i)/arma::normcdf(L20i),
               MU(1, i) - sqrt(a22) * arma::normpdf(L20i)/arma::normcdf(L20i),
               MU(0, i)*MU(0, i) + a11 - (2*MU(0, i)*a12/sqrt(a22) + a12*a12/a22*L20i) * arma::normpdf(L20i)/arma::normcdf(L20i),
               MU(0, i)*MU(1, i) + a12 - (MU(0, i)*sqrt(a22) + MU(1, i)*a12/sqrt(a22) + a12*L20i) * arma::normpdf(L20i)/arma::normcdf(L20i),
               MU(1, i)*MU(1, i) + a22 - (2*MU(1, i)*sqrt(a22) + a22*L20i) * arma::normpdf(L20i)/arma::normcdf(L20i)};
        Ehat.row(i) = vec;
      }
      else if(R2(i) == 2){
        U20i = (-S2(i) + MU(1, i)) / sqrt(a22);
        vec = {MU(0, i) + a12 / sqrt(a22) * arma::normpdf(U20i)/arma::normcdf(U20i),
               MU(1, i) + sqrt(a22) * arma::normpdf(U20i)/arma::normcdf(U20i),
               MU(0, i)*MU(0, i) + a11 + (2*MU(0, i)*a12/sqrt(a22) - a12*a12/a22*U20i) * arma::normpdf(U20i)/arma::normcdf(U20i),
               MU(0, i)*MU(1, i) + a12 + (MU(0, i)*sqrt(a22) + MU(1, i)*a12/sqrt(a22) - a12*U20i) * arma::normpdf(U20i)/arma::normcdf(U20i),
               MU(1, i)*MU(1, i) + a22 + (2*MU(1, i)*sqrt(a22) - a22*U20i) * arma::normpdf(U20i)/arma::normcdf(U20i)};
        Ehat.row(i) = vec;
      }
      else{
        stop("Inadmissible values: elements of vector R2 should be chosen from (0, 1, 2, 3).");
      }
    }
    else if(R1(i) == 1){
      if(R2(i) == 0){
        mu1giv2i = MU(0, i) + (S2(i) - MU(1, i)) * a12 / a22;
        L1giv2i = (S1(i) - mu1giv2i) / sqrt(a1giv2);
        vec = {mu1giv2i - sqrt(a1giv2) * arma::normpdf(L1giv2i)/arma::normcdf(L1giv2i), S2(i),
               mu1giv2i*mu1giv2i + a1giv2 - (2*mu1giv2i*sqrt(a1giv2) + a1giv2*L1giv2i) * arma::normpdf(L1giv2i)/arma::normcdf(L1giv2i),
               S2(i) * (mu1giv2i - sqrt(a1giv2) * arma::normpdf(L1giv2i)/arma::normcdf(L1giv2i)), S2(i) * S2(i)};
        Ehat.row(i) = vec;
      }
      else if(R2(i) == 3){
        L10i = (S1(i) - MU(0, i)) / sqrt(a11);
        vec = {MU(0, i) - sqrt(a11) * arma::normpdf(L10i)/arma::normcdf(L10i),
               MU(1, i) - a12 / sqrt(a11) * arma::normpdf(L10i)/arma::normcdf(L10i),
               MU(0, i)*MU(0, i) + a11 - (2*MU(0, i)*sqrt(a11) + a11*L10i) * arma::normpdf(L10i)/arma::normcdf(L10i),
               MU(0, i)*MU(1, i) + a12 - (MU(0, i)*a12/sqrt(a11) + MU(1, i)*sqrt(a11) + a12*L10i) * arma::normpdf(L10i)/arma::normcdf(L10i),
               MU(1, i)*MU(1, i) + a22 - (2*MU(1, i)*a12/sqrt(a11) + a12*a12/a11*L10i) * arma::normpdf(L10i)/arma::normcdf(L10i)};
        Ehat.row(i) = vec;
      }
      else if(R2(i) == 1){
        // Preparations
        I0 = 0; I1 = 0; I2 = 0; I3 = 0; I4 = 0; I5 = 0;
        L20i = (S2(i) - MU(1, i)) / sqrt(a22);
        tilde_mu1giv2i = MU(0, i) * arma::ones(K) + sqrt(2)*a12/sqrt(a22) * point_s;
        xi1i = (S1(i) * arma::ones(K) - tilde_mu1giv2i) / sqrt(a1giv2);
        
        // Find the index that satisfies s[index - 1] <= L20i/sqrt(2) && s[index] > L20i/sqrt(2)
        index = find_largest_index(point_s, L20i/sqrt(2));
        
        // Computations of the sum
        for(int k=0; k<index; k++){
          I0 = I0 + weight_w(k) * arma::normcdf(xi1i(k));
          I1 = I1 + weight_w(k) * (tilde_mu1giv2i(k) * arma::normcdf(xi1i(k)) - sqrt(a1giv2) * arma::normpdf(xi1i(k)));
          I2 = I2 + weight_w(k) * (MU(1, i) + sqrt(2 * a22) * point_s(k)) * arma::normcdf(xi1i(k));
          I3 = I3 + weight_w(k) * ((tilde_mu1giv2i(k)*tilde_mu1giv2i(k) + a1giv2) * arma::normcdf(xi1i(k))
                                     - (2*tilde_mu1giv2i(k)*sqrt(a1giv2) + a1giv2*xi1i(k)) * arma::normpdf(xi1i(k)));
          I4 = I4 + weight_w(k) * (MU(1, i) + sqrt(2 * a22) * point_s(k)) * (tilde_mu1giv2i(k) * arma::normcdf(xi1i(k)) - sqrt(a1giv2) * arma::normpdf(xi1i(k)));
          I5 = I5 + weight_w(k) * pow(MU(1, i) + sqrt(2 * a22) * point_s(k), 2) * arma::normcdf(xi1i(k));
        }
        vec = {I1/I0, I2/I0, I3/I0, I4/I0, I5/I0};
        Ehat.row(i) = vec;
      }
      else if(R2(i) == 2){
        // Preparations
        I0 = 0; I1 = 0; I2 = 0; I3 = 0; I4 = 0; I5 = 0;
        U20i = (-S2(i) + MU(1, i)) / sqrt(a22);
        tilde_mu1giv2i = MU(0, i) * arma::ones(K) + sqrt(2)*a12/sqrt(a22) * point_s;
        xi1i = (S1(i) * arma::ones(K) - tilde_mu1giv2i) / sqrt(a1giv2);
        
        // Find the index that satisfies s[index - 1] < -U20i/sqrt(2) && s[index] >= -U20i/sqrt(2)
        index = find_smallest_index(point_s, -U20i/sqrt(2));
        
        // Computations of the sum
        for(int k=index; k<K; k++){
          I0 = I0 + weight_w(k) * arma::normcdf(xi1i(k));
          I1 = I1 + weight_w(k) * (tilde_mu1giv2i(k) * arma::normcdf(xi1i(k)) - sqrt(a1giv2) * arma::normpdf(xi1i(k)));
          I2 = I2 + weight_w(k) * (MU(1, i) + sqrt(2 * a22) * point_s(k)) * arma::normcdf(xi1i(k));
          I3 = I3 + weight_w(k) * ( ( tilde_mu1giv2i(k)*tilde_mu1giv2i(k) + a1giv2) * arma::normcdf(xi1i(k))
                                      - (2*tilde_mu1giv2i(k)*sqrt(a1giv2) + a1giv2*xi1i(k)) * arma::normpdf(xi1i(k)) );
          I4 = I4 + weight_w(k) * (MU(1, i) + sqrt(2 * a22) * point_s(k)) * (tilde_mu1giv2i(k) * arma::normcdf(xi1i(k)) - sqrt(a1giv2) * arma::normpdf(xi1i(k)));
          I5 = I5 + weight_w(k) * pow(MU(1, i) + sqrt(2 * a22) * point_s(k), 2) * arma::normcdf(xi1i(k));
        }
        vec = {I1/I0, I2/I0, I3/I0, I4/I0, I5/I0};
        Ehat.row(i) = vec;
      }
      else{
        stop("Inadmissible values: elements of vector R2 should be chosen from (0, 1, 2, 3).");
      }
    }
    else if(R1(i) == 2){
      if(R2(i) == 0){
        mu1giv2i = MU(0, i) + (S2(i) - MU(1, i)) * a12 / a22;
        U1giv2i = (-S1(i) + mu1giv2i) / sqrt(a1giv2);
        vec = {mu1giv2i + sqrt(a1giv2) * arma::normpdf(U1giv2i)/arma::normcdf(U1giv2i), S2(i),
               mu1giv2i*mu1giv2i + a1giv2 + (2*mu1giv2i*sqrt(a1giv2) - a1giv2*U1giv2i) * arma::normpdf(U1giv2i)/arma::normcdf(U1giv2i),
               S2(i) * (mu1giv2i + sqrt(a1giv2) * arma::normpdf(U1giv2i)/arma::normcdf(U1giv2i)), S2(i) * S2(i)};
        Ehat.row(i) = vec;
      }
      else if(R2(i) == 3){
        U10i = (-S1(i) + MU(0, i)) / sqrt(a11);
        vec = {MU(0, i) + sqrt(a11) * arma::normpdf(U10i)/arma::normcdf(U10i),
               MU(1, i) + a12 / sqrt(a11) * arma::normpdf(U10i)/arma::normcdf(U10i),
               MU(0, i)*MU(0, i) + a11 + (2*MU(0, i)*sqrt(a11) - a11*U10i) * arma::normpdf(U10i)/arma::normcdf(U10i),
               MU(0, i)*MU(1, i) + a12 + (MU(0, i)*a12/sqrt(a11) + MU(1, i)*sqrt(a11) - a12*U10i) * arma::normpdf(U10i)/arma::normcdf(U10i),
               MU(1, i)*MU(1, i) + a22 + (2*MU(1, i)*a12/sqrt(a11) - a12*a12/a11*U10i) * arma::normpdf(U10i)/arma::normcdf(U10i)};
        Ehat.row(i) = vec;
      }
      else if(R2(i) == 1){
        // Preparations
        I0 = 0; I1 = 0; I2 = 0; I3 = 0; I4 = 0; I5 = 0;
        L20i = (S2(i) - MU(1, i)) / sqrt(a22);
        tilde_mu1giv2i = MU(0, i) * arma::ones(K) + sqrt(2)*a12/sqrt(a22) * point_s;
        xi2i = (-S1(i) * arma::ones(K) + tilde_mu1giv2i) / sqrt(a1giv2);
        
        // Find the index that satisfies s[index - 1] <= L20i/sqrt(2) && s[index] > L20i/sqrt(2)
        index = find_largest_index(point_s, L20i/sqrt(2));
        
        // Computations of the sum
        for(int k=0; k<index; k++){
          I0 = I0 + weight_w(k) * arma::normcdf(xi2i(k));
          I1 = I1 + weight_w(k) * (tilde_mu1giv2i(k) * arma::normcdf(xi2i(k)) + sqrt(a1giv2) * arma::normpdf(xi2i(k)));
          I2 = I2 + weight_w(k) * (MU(1, i) + sqrt(2 * a22) * point_s(k)) * arma::normcdf(xi2i(k));
          I3 = I3 + weight_w(k) * ((tilde_mu1giv2i(k)*tilde_mu1giv2i(k) + a1giv2) * arma::normcdf(xi2i(k))
                                     + (2*tilde_mu1giv2i(k)*sqrt(a1giv2) - a1giv2*xi2i(k)) * arma::normpdf(xi2i(k)));
          I4 = I4 + weight_w(k) * (MU(1, i) + sqrt(2 * a22) * point_s(k)) * (tilde_mu1giv2i(k) * arma::normcdf(xi2i(k)) + sqrt(a1giv2) * arma::normpdf(xi2i(k)));
          I5 = I5 + weight_w(k) * pow(MU(1, i) + sqrt(2 * a22) * point_s(k), 2) * arma::normcdf(xi2i(k));
        }
        vec = {I1/I0, I2/I0, I3/I0, I4/I0, I5/I0};
        Ehat.row(i) = vec;
      }
      else if(R2(i) == 2){
        // Preparations
        I0 = 0; I1 = 0; I2 = 0; I3 = 0; I4 = 0; I5 = 0;
        U20i = (-S2(i) + MU(1, i)) / sqrt(a22);
        tilde_mu1giv2i = MU(0, i) * arma::ones(K) + sqrt(2)*a12/sqrt(a22) * point_s;
        xi2i = (-S1(i) * arma::ones(K) + tilde_mu1giv2i) / sqrt(a1giv2);
        
        // Find the index that satisfies s[index - 1] < -U20i/sqrt(2) && s[index] >= -U20i/sqrt(2)
        index = find_smallest_index(point_s, -U20i/sqrt(2));
        
        // Computations of the sum
        for(int k=index; k<K; k++){
          I0 = I0 + weight_w(k) * arma::normcdf(xi2i(k));
          I1 = I1 + weight_w(k) * (tilde_mu1giv2i(k) * arma::normcdf(xi2i(k)) + sqrt(a1giv2) * arma::normpdf(xi2i(k)));
          I2 = I2 + weight_w(k) * (MU(1, i) + sqrt(2 * a22) * point_s(k)) * arma::normcdf(xi2i(k));
          I3 = I3 + weight_w(k) * ((tilde_mu1giv2i(k)*tilde_mu1giv2i(k) + a1giv2) * arma::normcdf(xi2i(k))
                                     + (2*tilde_mu1giv2i(k)*sqrt(a1giv2) - a1giv2*xi2i(k)) * arma::normpdf(xi2i(k)));
          I4 = I4 + weight_w(k) * (MU(1, i) + sqrt(2 * a22) * point_s(k)) * (tilde_mu1giv2i(k) * arma::normcdf(xi2i(k)) + sqrt(a1giv2) * arma::normpdf(xi2i(k)));
          I5 = I5 + weight_w(k) * pow(MU(1, i) + sqrt(2 * a22) * point_s(k), 2) * arma::normcdf(xi2i(k));
        }
        vec = {I1/I0, I2/I0, I3/I0, I4/I0, I5/I0};
        Ehat.row(i) = vec;
      }
      else{
        stop("Inadmissible values: elements of vector R2 should be chosen from (0, 1, 2, 3).");
      }
    }
    else{
      stop("Inadmissible values: elements of vector R1 should be chosen from (0, 1, 2, 3).");
    }
  }
  //cout << "3-1-4" << endl;
  
  return Ehat;
}



// M-step
struct MatDetGradHessian{
  arma::mat mat;
  double det;
  arma::vec grad;
  arma::mat hessian;
};


struct para{
  arma::vec alpha1;
  arma::vec alpha2;
  arma::vec beta;
  double gamma1;
  double gamma2;
  arma::vec longpara;
  arma::mat Sigma;
  int m;
};


//// Profile log-likelihood
struct MatDetGradHessian ProfileLoglik(arma::mat X, arma::vec Y, arma::mat Z, arma::mat Ehat,
                                       arma::vec alpha1, arma::vec alpha2, arma::vec beta, double gamma1, double gamma2){
  struct MatDetGradHessian Qresult;
  int n = Y.n_elem;
  arma::vec E1 = Ehat.col(0);
  arma::vec E2 = Ehat.col(1);
  double E11 = arma::sum(Ehat.col(2));
  double E12 = arma::sum(Ehat.col(3));
  double E22 = arma::sum(Ehat.col(4));
  
  ////// 1) Definition of Qij and their non-zero 1st & 2nd derivatives:
  double Q11 = (arma::as_scalar((E1 - X*alpha1).t() * (E1 - X*alpha1)) + E11 - arma::as_scalar(E1.t() * E1)) / n;
  double Q12 = (arma::as_scalar((E1 - X*alpha1).t() * (E2 - X*alpha2)) + E12 - arma::as_scalar(E1.t() * E2)) / n;
  double Q13 = (arma::as_scalar((E1 - X*alpha1).t() * (Y - Z*beta - gamma1*E1 - gamma2*E2)) 
                  - gamma1 * (E11 - arma::as_scalar(E1.t() * E1)) - gamma2 * (E12 - arma::as_scalar(E1.t() * E2))) / n;
  double Q22 = (arma::as_scalar((E2 - X*alpha2).t() * (E2 - X*alpha2)) + E22 - arma::as_scalar(E2.t() * E2)) / n;
  double Q23 = (arma::as_scalar((E2 - X*alpha2).t() * (Y - Z*beta - gamma1*E1 - gamma2*E2)) 
                  - gamma1 * (E12 - arma::as_scalar(E1.t() * E2)) - gamma2 * (E22 - arma::as_scalar(E2.t() * E2))) / n;
  double Q33 = (arma::as_scalar((Y - Z*beta - gamma1*E1 - gamma2*E2).t() * (Y - Z*beta - gamma1*E1 - gamma2*E2))
                  + gamma1*gamma1 * (E11 - arma::as_scalar(E1.t() * E1))
                  + 2*gamma1*gamma2 * (E12 - arma::as_scalar(E1.t() * E2))
                  + gamma2*gamma2 * (E22 - arma::as_scalar(E2.t() * E2))) / n;
  
  arma::vec Q11_alpha1 = 2 * X.t() * (X*alpha1 - E1) / n;
  arma::mat Q11_alpha1_alpha1 = 2 * X.t() * X / n;
  //Rcout << Q11_alpha1 << endl;
    
  arma::vec Q12_alpha1 = X.t() * (X*alpha2 - E2) / n;
  arma::vec Q12_alpha2 = X.t() * (X*alpha1 - E1) / n;
  arma::mat Q12_alpha1_alpha2 = X.t() * X / n;
  //Rcout << Q12_alpha1 << endl;
  //Rcout << Q12_alpha2 << endl;
  
  arma::vec Q22_alpha2 = 2 * X.t() * (X*alpha2 - E2) / n;
  arma::mat Q22_alpha2_alpha2 = 2 * X.t() * X / n;
  //Rcout << Q22_alpha2 << endl;
  
  arma::vec Q13_alpha1 = X.t() * (Z*beta + gamma1*E1 + gamma2*E2 - Y) / n;
  arma::vec Q13_beta = Z.t() * (X*alpha1 - E1) / n;
  double Q13_gamma1 = (arma::as_scalar(E1.t()*X*alpha1) - E11) / n;
  double Q13_gamma2 = (arma::as_scalar(E2.t()*X*alpha1) - E12) / n;
  arma::mat Q13_alpha1_beta = X.t() * Z / n;
  arma::vec Q13_alpha1_gamma1 = X.t() * E1 / n;
  arma::vec Q13_alpha1_gamma2 = X.t() * E2 / n;
  //Rcout << Q13_alpha1 << endl;
  //Rcout << Q13_beta << endl;
  //Rcout << Q13_gamma1 << endl;
  //Rcout << Q13_gamma2 << endl;
  
  arma::vec Q23_alpha2 = X.t() * (Z*beta + gamma1*E1 + gamma2*E2 - Y) / n;
  arma::vec Q23_beta = Z.t() * (X*alpha2 - E2) / n;
  double Q23_gamma1 = (arma::as_scalar(E1.t()*X*alpha2) - E12) / n;
  double Q23_gamma2 = (arma::as_scalar(E2.t()*X*alpha2) - E22) / n;
  arma::mat Q23_alpha2_beta = X.t() * Z / n;
  arma::vec Q23_alpha2_gamma1 = X.t() * E1 / n;
  arma::vec Q23_alpha2_gamma2 = X.t() * E2 / n;
  //Rcout << Q23_alpha2 << endl;
  //Rcout << Q23_beta << endl;
  //Rcout << Q23_gamma1 << endl;
  //Rcout << Q23_gamma2 << endl;
  
  arma::vec Q33_beta = 2 * Z.t() * (Z*beta + gamma1*E1 + gamma2*E2 - Y) / n;
  double Q33_gamma1 = 2 * (arma::as_scalar(beta.t()*Z.t()*E1) + gamma1*E11 + gamma2*E12 - arma::as_scalar(Y.t()*E1) ) / n;
  double Q33_gamma2 = 2 * (arma::as_scalar(beta.t()*Z.t()*E2) + gamma1*E12 + gamma2*E22 - arma::as_scalar(Y.t()*E2) ) / n;
  arma::mat Q33_beta_beta = 2 * Z.t() * Z / n;
  arma::vec Q33_beta_gamma1 = 2 * Z.t() * E1 / n;
  arma::vec Q33_beta_gamma2 = 2 * Z.t() * E2 / n;
  double Q33_gamma1_gamma1 = 2 * E11 / n;
  double Q33_gamma1_gamma2 = 2 * E12 / n;
  double Q33_gamma2_gamma2 = 2 * E22 / n;
  //Rcout << Q33_beta << endl;
  //Rcout << Q33_gamma1 << endl;
  //Rcout << Q33_gamma2 << endl;
  //cout << "3-2-1" << endl;
  
  ////// 2) Gradient (1st derivative) of function Q 
  arma::vec Q_alpha1 = Q11_alpha1*Q22*Q33 + 2*Q23 * (Q12_alpha1*Q13 + Q12*Q13_alpha1) - Q11_alpha1*Q23*Q23 - 2*Q22*Q13*Q13_alpha1 - 2*Q33*Q12*Q12_alpha1;
  arma::vec Q_alpha2 = Q11*Q22_alpha2*Q33 + 2*Q13 * (Q12_alpha2*Q23 + Q12*Q23_alpha2) - 2*Q11*Q23*Q23_alpha2 - Q22_alpha2*Q13*Q13 - 2*Q33*Q12*Q12_alpha2;
  arma::vec Q_beta = Q11*Q22*Q33_beta + 2*Q12 * (Q13_beta*Q23 + Q13*Q23_beta) - 2*Q11*Q23*Q23_beta - 2*Q22*Q13*Q13_beta - Q33_beta*Q12*Q12;
  arma::vec Q_gamma = {Q11*Q22*Q33_gamma1 + 2*Q12 * (Q13_gamma1*Q23 + Q13*Q23_gamma1) - 2*Q11*Q23*Q23_gamma1 - 2*Q22*Q13*Q13_gamma1 - Q33_gamma1*Q12*Q12,
                       Q11*Q22*Q33_gamma2 + 2*Q12 * (Q13_gamma2*Q23 + Q13*Q23_gamma2) - 2*Q11*Q23*Q23_gamma2 - 2*Q22*Q13*Q13_gamma2 - Q33_gamma2*Q12*Q12};
  arma::vec Grad = arma::join_cols(arma::join_cols(Q_alpha1, Q_alpha2), arma::join_cols(Q_beta, Q_gamma));
  //cout << "3-2-2" << endl;
  
  ////// 3) Hessian Matrix (2nd derivatives) of function Q
  ////// alpha1
  arma::mat Q_alpha1_alpha1 = Q11_alpha1_alpha1 * (Q22*Q33 - Q23*Q23) + 2 * Q23 * (Q13_alpha1*Q12_alpha1.t() + Q12_alpha1*Q13_alpha1.t())
    - 2*Q22*Q13_alpha1*Q13_alpha1.t() - 2*Q33*Q12_alpha1*Q12_alpha1.t();
  
  arma::mat Q_alpha1_alpha2 = 2 * Q12_alpha1_alpha2 * (Q13*Q23 - Q33*Q12) + 2 * Q12 * Q13_alpha1 * Q23_alpha2.t() 
    + Q33 * (Q11_alpha1*Q22_alpha2.t() - 2*Q12_alpha1*Q12_alpha2.t()) 
    + 2 * Q23 * (Q13_alpha1*Q12_alpha2.t() - Q11_alpha1*Q23_alpha2.t()) + 2 * Q13 * (Q12_alpha1*Q23_alpha2.t() - Q13_alpha1*Q22_alpha2.t());
    
  arma::mat Q_alpha1_beta = 2 * Q13_alpha1_beta * (Q12*Q23 - Q22*Q13) + 2 * Q13 * Q12_alpha1 * Q23_beta.t() 
    + Q22 * (Q11_alpha1*Q33_beta.t() - 2*Q13_alpha1*Q13_beta.t()) 
    + 2 * Q23 * (Q12_alpha1*Q13_beta.t() - Q11_alpha1*Q23_beta.t()) + 2 * Q12 * (Q13_alpha1*Q23_beta.t() - Q12_alpha1*Q33_beta.t());
      
  arma::vec Q_alpha1_gamma1 = 2 * Q13_alpha1_gamma1 * (Q12*Q23 - Q22*Q13) + 2 * Q13 * Q12_alpha1 * Q23_gamma1
    + Q22 * (Q11_alpha1*Q33_gamma1 - 2*Q13_alpha1*Q13_gamma1) 
    + 2 * Q23 * (Q12_alpha1*Q13_gamma1 - Q11_alpha1*Q23_gamma1) + 2 * Q12 * (Q13_alpha1*Q23_gamma1 - Q12_alpha1*Q33_gamma1);
        
  arma::vec Q_alpha1_gamma2 = 2 * Q13_alpha1_gamma2 * (Q12*Q23 - Q22*Q13) + 2 * Q13 * Q12_alpha1 * Q23_gamma2
    + Q22 * (Q11_alpha1*Q33_gamma2 - 2*Q13_alpha1*Q13_gamma2) 
    + 2 * Q23 * (Q12_alpha1*Q13_gamma2 - Q11_alpha1*Q23_gamma2) + 2 * Q12 * (Q13_alpha1*Q23_gamma2 - Q12_alpha1*Q33_gamma2);
  //cout << "3-2-3-1" << endl;
          
  ////// alpha2
  arma::mat Q_alpha2_alpha2 = Q22_alpha2_alpha2 * (Q11*Q33 - Q13*Q13) + 2 * Q13 * (Q23_alpha2*Q12_alpha2.t() + Q12_alpha2*Q23_alpha2.t())
    - 2*Q11*Q23_alpha2*Q23_alpha2.t() - 2*Q33*Q12_alpha2*Q12_alpha2.t();
          
  arma::mat Q_alpha2_beta = 2 * Q23_alpha2_beta * (Q12*Q13 - Q11*Q23) + 2 * Q23 * Q12_alpha2 * Q13_beta.t() 
    + Q11 * (Q22_alpha2*Q33_beta.t() - 2*Q23_alpha2*Q23_beta.t()) 
    + 2 * Q13 * (Q12_alpha2*Q23_beta.t() - Q22_alpha2*Q13_beta.t()) + 2 * Q12 * (Q23_alpha2*Q13_beta.t() - Q12_alpha2*Q33_beta.t());
            
  arma::vec Q_alpha2_gamma1 = 2 * Q23_alpha2_gamma1 * (Q12*Q13 - Q11*Q23) + 2 * Q23 * Q12_alpha2 * Q13_gamma1 
    + Q11 * (Q22_alpha2*Q33_gamma1 - 2*Q23_alpha2*Q23_gamma1) 
    + 2 * Q13 * (Q12_alpha2*Q23_gamma1 - Q22_alpha2*Q13_gamma1) + 2 * Q12 * (Q23_alpha2*Q13_gamma1 - Q12_alpha2*Q33_gamma1);
              
  arma::vec Q_alpha2_gamma2 = 2 * Q23_alpha2_gamma2 * (Q12*Q13 - Q11*Q23) + 2 * Q23 * Q12_alpha2 * Q13_gamma2 
    + Q11 * (Q22_alpha2*Q33_gamma2 - 2*Q23_alpha2*Q23_gamma2) 
    + 2 * Q13 * (Q12_alpha2*Q23_gamma2 - Q22_alpha2*Q13_gamma2) + 2 * Q12 * (Q23_alpha2*Q13_gamma2 - Q12_alpha2*Q33_gamma2);
  //cout << "3-2-3-2" << endl;
                
  ////// beta
  arma::mat Q_beta_beta = Q33_beta_beta * (Q11*Q22 - Q12*Q12) + 2 * Q12 * (Q23_beta*Q13_beta.t() + Q13_beta*Q23_beta.t())
    - 2*Q11*Q23_beta*Q23_beta.t() - 2*Q22*Q13_beta*Q13_beta.t();
                
  arma::vec Q_beta_gamma1 = Q33_beta_gamma1 * (Q11*Q22 - Q12*Q12) + 2 * Q12 * (Q23_beta*Q13_gamma1 + Q13_beta*Q23_gamma1)
    - 2*Q11*Q23_beta*Q23_gamma1 - 2*Q22*Q13_beta*Q13_gamma1;
                
  arma::vec Q_beta_gamma2 = Q33_beta_gamma2 * (Q11*Q22 - Q12*Q12) + 2 * Q12 * (Q23_beta*Q13_gamma2 + Q13_beta*Q23_gamma2)
    - 2*Q11*Q23_beta*Q23_gamma2 - 2*Q22*Q13_beta*Q13_gamma2;
  //cout << "3-2-3-3" << endl;
                
  ////// gamma1
  double Q_gamma1_gamma1_numeric = Q33_gamma1_gamma1 * (Q11*Q22 - Q12*Q12) + 4 * Q12 * Q13_gamma1 * Q23_gamma1 - 2*Q11*Q23_gamma1*Q23_gamma1 - 2*Q22*Q13_gamma1*Q13_gamma1;
  double Q_gamma1_gamma2_numeric = Q33_gamma1_gamma2 * (Q11*Q22 - Q12*Q12) + 2 * Q12 * (Q23_gamma1*Q13_gamma2 + Q13_gamma1*Q23_gamma2)
    - 2*Q11*Q23_gamma1*Q23_gamma2 - 2*Q22*Q13_gamma1*Q13_gamma2;
  arma::mat Q_gamma1_gamma1 = arma::zeros(1, 1), Q_gamma1_gamma2 = arma::zeros(1, 1);
  Q_gamma1_gamma1(0, 0) = Q_gamma1_gamma1_numeric;
  Q_gamma1_gamma2(0, 0) = Q_gamma1_gamma2_numeric;
  //cout << "3-2-3-4" << endl;
                
  ////// gamma2
  double Q_gamma2_gamma2_numeric = Q33_gamma2_gamma2 * (Q11*Q22 - Q12*Q12) + 4 * Q12 * Q13_gamma2 * Q23_gamma2 - 2*Q11*Q23_gamma2*Q23_gamma2 - 2*Q22*Q13_gamma2*Q13_gamma2;
  arma::mat Q_gamma2_gamma2 = arma::zeros(1, 1);
  Q_gamma2_gamma2(0, 0) = Q_gamma2_gamma2_numeric;
  //cout << "3-2-3-5" << endl;
                
  ////// Hessian Matrix
  arma::mat HessianMat = 
    arma::join_cols(
      arma::join_cols(
        arma::join_rows(arma::join_rows(Q_alpha1_alpha1, Q_alpha1_alpha2), arma::join_rows(Q_alpha1_beta, arma::join_rows(Q_alpha1_gamma1, Q_alpha1_gamma2))),
        arma::join_cols(arma::join_rows(arma::join_rows(Q_alpha1_alpha2.t(), Q_alpha2_alpha2), arma::join_rows(Q_alpha2_beta, arma::join_rows(Q_alpha2_gamma1, Q_alpha2_gamma2))),
        arma::join_rows(arma::join_rows(Q_alpha1_beta.t(), Q_alpha2_beta.t()), arma::join_rows(Q_beta_beta, arma::join_rows(Q_beta_gamma1, Q_beta_gamma2))))),
      arma::join_cols(
        arma::join_rows(arma::join_rows(Q_alpha1_gamma1.t(), Q_alpha2_gamma1.t()), arma::join_rows(Q_beta_gamma1.t(), arma::join_rows(Q_gamma1_gamma1, Q_gamma1_gamma2))),
        arma::join_rows(arma::join_rows(Q_alpha1_gamma2.t(), Q_alpha2_gamma2.t()), arma::join_rows(Q_beta_gamma2.t(), arma::join_rows(Q_gamma1_gamma2, Q_gamma2_gamma2))))
    );
  //cout << "3-2-3-6" << endl;
  
  arma::mat Q = { {Q11, Q12, Q13}, {Q12, Q22, Q23}, {Q13, Q23, Q33} };
  Qresult.mat = Q;        // The maximizer Q = \hat\Sigma when alpha1, alpha2, beta, gamma1, gamma2 are fixed
  Qresult.det = det(Q);   // The equivalent profile log-likelihood to be minimized
  Qresult.grad = Grad;
  Qresult.hessian = HessianMat;
  //Rcout << Q << endl;
  //Rcout << det(Q) << endl;
  //Rcout << Grad << endl;
  //Rcout << HessianMat << endl;
  
  return Qresult;
}



//// Regression parameters (alpha1, alpha2, beta, gamma1, gamma2) and variance component (Sigma) update
struct para RegPara_Varcomp_update(arma::mat X, arma::vec Y, arma::mat Z, arma::mat Ehat, 
                                   arma::vec alpha1, arma::vec alpha2, arma::vec beta, double gamma1, double gamma2, double epsilon){
  struct para para_est;
  arma::vec alpha1_update, alpha2_update, beta_update;
  double gamma1_update = 0, gamma2_update = 0;
  
  ////// 1) Step size & Regression parameters update
  //////// Preparations
  arma::vec gamma = arma::zeros(2);
  gamma(0) = gamma1;
  gamma(1) = gamma2;
  arma::vec para_current = arma::join_cols(arma::join_cols(alpha1, alpha2), arma::join_cols(beta, gamma));
  arma::vec para_update = para_current;
  struct MatDetGradHessian Q_current = ProfileLoglik(X, Y, Z, Ehat, alpha1, alpha2, beta, gamma1, gamma2);
  
  //////// Parameters Update
  //arma::vec stepsize = (Q_current.hessian - Q_current.grad * Q_current.grad.t() / (Q_current.det + 10)).i() * Q_current.grad;
  arma::vec stepsize = Q_current.hessian.i() * Q_current.grad;
  para_update = para_current - stepsize;
  
  alpha1_update = para_update.subvec(0, alpha1.n_elem - 1);
  alpha2_update = para_update.subvec(alpha1.n_elem, alpha1.n_elem + alpha2.n_elem - 1);
  beta_update = para_update.subvec(alpha1.n_elem + alpha2.n_elem, alpha1.n_elem + alpha2.n_elem + beta.n_elem - 1);
  gamma1_update = para_update(alpha1.n_elem + alpha2.n_elem + beta.n_elem);
  gamma2_update = para_update(alpha1.n_elem + alpha2.n_elem + beta.n_elem + 1);
  struct MatDetGradHessian Q_update = ProfileLoglik(X, Y, Z, Ehat, alpha1_update, alpha2_update, beta_update, gamma1_update, gamma2_update);
  
  int m = 0;
  while(Q_update.det > Q_current.det){
    m = m + 1;
    para_update = para_current - stepsize / pow(2, m);
    
    alpha1_update = para_update.subvec(0, alpha1.n_elem - 1);
    alpha2_update = para_update.subvec(alpha1.n_elem, alpha1.n_elem + alpha2.n_elem - 1);
    beta_update = para_update.subvec(alpha1.n_elem + alpha2.n_elem, alpha1.n_elem + alpha2.n_elem + beta.n_elem - 1);
    gamma1_update = para_update(alpha1.n_elem + alpha2.n_elem + beta.n_elem);
    gamma2_update = para_update(alpha1.n_elem + alpha2.n_elem + beta.n_elem + 1);
    Q_update = ProfileLoglik(X, Y, Z, Ehat, alpha1_update, alpha2_update, beta_update, gamma1_update, gamma2_update);
  }
  //cout << "m = " << m << " ";
  
  para_est.alpha1 = alpha1_update;
  para_est.alpha2 = alpha2_update;
  para_est.beta = beta_update;
  para_est.gamma1 = gamma1_update;
  para_est.gamma2 = gamma2_update;
  para_est.longpara = para_update;
  
  ////// 2) Variance components update
  para_est.Sigma = Q_update.mat;
  para_est.m = m;
                
  return para_est;
}



// Louis Formula: Computaion of Ic
arma::mat Info_complete(arma::mat X, arma::vec Y, arma::mat Z, arma::mat Ehat,
                        arma::vec alpha1, arma::vec alpha2, arma::vec beta, double gamma1, double gamma2, arma::mat Sigma){
  int n = Y.n_elem;
  int a = X.n_cols;
  int b = Z.n_cols;
  arma::mat Ic = arma::zeros(a+a+b+8, a+a+b+8);
  
  double sigma1square = Sigma(0, 0);
  double sigma1 = sqrt(sigma1square);
  double sigma2square = Sigma(1, 1);
  double sigma2 = sqrt(sigma2square);
  double sigmaYsquare = Sigma(2, 2);
  double sigmaY = sqrt(sigmaYsquare);
  double rho12 = Sigma(0, 1) / (sigma1 * sigma2);
  double rho1Y = Sigma(0, 2) / (sigma1 * sigmaY);
  double rho2Y = Sigma(1, 2) / (sigma2 * sigmaY);
  
  arma::vec E1 = Ehat.col(0);
  arma::vec E2 = Ehat.col(1);
  arma::vec Res1 = E1 - X * alpha1;
  arma::vec Res2 = E2 - X * alpha2;
  arma::vec ResY = Y - Z * beta - gamma1 * E1 - gamma2 * E2;
  //double C11 = sum(Ehat.col(2)) - arma::as_scalar(E1.t() * E1);
  //double C12 = sum(Ehat.col(3)) - arma::as_scalar(E1.t() * E2);
  //double C22 = sum(Ehat.col(4)) - arma::as_scalar(E2.t() * E2);
  double Crho = 1 + 2*rho12*rho1Y*rho2Y - rho12*rho12 - rho1Y*rho1Y - rho2Y*rho2Y;
  double rhoconst11 = (1 - rho2Y*rho2Y) / sigma1square;
  double rhoconst22 = (1 - rho1Y*rho1Y) / sigma2square;
  double rhoconstYY = (1 - rho12*rho12) / sigmaYsquare;
  double rhoconst12 = (rho1Y*rho2Y - rho12) / (sigma1*sigma2);
  double rhoconst1Y = (rho12*rho2Y - rho1Y) / (sigma1*sigmaY);
  double rhoconst2Y = (rho12*rho1Y - rho2Y) / (sigma2*sigmaY);
  
  arma::vec XtRes1 = X.t() * Res1;
  arma::vec XtRes2 = X.t() * Res2;
  arma::vec XtResY = X.t() * ResY;
  arma::vec ZtRes1 = Z.t() * Res1;
  arma::vec ZtRes2 = Z.t() * Res2;
  arma::vec ZtResY = Z.t() * ResY;
  double E1tRes1 = sum(Ehat.col(2)) - arma::as_scalar(E1.t() * X * alpha1);
  double E1tRes2 = sum(Ehat.col(3)) - arma::as_scalar(E1.t() * X * alpha2);
  double E1tResY = arma::as_scalar(E1.t() * (Y - Z*beta)) - gamma1 * sum(Ehat.col(2)) - gamma2 * sum(Ehat.col(3));
  double E2tRes1 = sum(Ehat.col(3)) - arma::as_scalar(E2.t() * X * alpha1);
  double E2tRes2 = sum(Ehat.col(4)) - arma::as_scalar(E2.t() * X * alpha2);
  double E2tResY = arma::as_scalar(E2.t() * (Y - Z*beta)) - gamma1 * sum(Ehat.col(3)) - gamma2 * sum(Ehat.col(4));
  
  //// alpha1
  //cout << "1-1" << endl;
  Ic(arma::span(0, a-1), arma::span(0, a-1)) = rhoconst11 / Crho * X.t() * X;
  Ic(arma::span(0, a-1), arma::span(a, a+a-1)) = rhoconst12 / Crho * X.t() * X;
  Ic(arma::span(0, a-1), arma::span(a+a, a+a+b-1)) = rhoconst1Y / Crho * X.t() * Z;
  Ic(arma::span(0, a-1), a+a+b) = rhoconst1Y / Crho * X.t() * E1;
  Ic(arma::span(0, a-1), a+a+b+1) = rhoconst1Y / Crho * X.t() * E2;
  Ic(arma::span(0, a-1), a+a+b+2) = 1 / (2*Crho) * (2*rhoconst11/sigma1square*XtRes1 + rhoconst12/sigma1square*XtRes2 + rhoconst1Y/sigma1square*XtResY);
  Ic(arma::span(0, a-1), a+a+b+3) = 1 / (2*Crho) * rhoconst12 / sigma2square * XtRes2;
  Ic(arma::span(0, a-1), a+a+b+4) = 1 / (2*Crho) * rhoconst1Y / sigmaYsquare * XtResY;
  Ic(arma::span(0, a-1), a+a+b+5) = 2*(rho1Y*rho2Y - rho12)/pow(Crho, 2) * (rhoconst11*XtRes1 + rhoconst12*XtRes2 + rhoconst1Y*XtResY) - 1/Crho * (-XtRes2/(sigma1*sigma2) + rho2Y*XtResY/(sigma1*sigmaY));
  Ic(arma::span(0, a-1), a+a+b+6) = 2*(rho12*rho2Y - rho1Y)/pow(Crho, 2) * (rhoconst11*XtRes1 + rhoconst12*XtRes2 + rhoconst1Y*XtResY) - 1/Crho * (rho2Y*XtRes2/(sigma1*sigma2) - XtResY/(sigma1*sigmaY));
  Ic(arma::span(0, a-1), a+a+b+7) = 2*(rho12*rho1Y - rho2Y)/pow(Crho, 2) * (rhoconst11*XtRes1 + rhoconst12*XtRes2 + rhoconst1Y*XtResY) - 1/Crho * (-2*rho2Y*XtRes1/sigma1square + rho1Y*XtRes2/(sigma1*sigma2) + rho12*XtResY/(sigma1*sigmaY));
  
  //// alpha2
  //cout << "1-2" << endl;
  Ic(arma::span(a, a+a-1), arma::span(0, a-1)) = Ic(arma::span(0, a-1), arma::span(a, a+a-1)).t();
  Ic(arma::span(a, a+a-1), arma::span(a, a+a-1)) = rhoconst22 / Crho * X.t() * X;
  Ic(arma::span(a, a+a-1), arma::span(a+a, a+a+b-1)) = rhoconst2Y / Crho * X.t() * Z;
  Ic(arma::span(a, a+a-1), a+a+b) = rhoconst2Y / Crho * X.t() * E1;
  Ic(arma::span(a, a+a-1), a+a+b+1) = rhoconst2Y / Crho * X.t() * E2;
  Ic(arma::span(a, a+a-1), a+a+b+2) = 1 / (2*Crho) * rhoconst12 / sigma1square * XtRes1;
  Ic(arma::span(a, a+a-1), a+a+b+3) = 1 / (2*Crho) * (2*rhoconst22/sigma2square*XtRes2 + rhoconst12/sigma2square * XtRes1 + rhoconst2Y/sigma2square*XtResY);
  Ic(arma::span(a, a+a-1), a+a+b+4) = 1 / (2*Crho) * rhoconst2Y / sigmaYsquare * XtResY;
  Ic(arma::span(a, a+a-1), a+a+b+5) = 2*(rho1Y*rho2Y - rho12)/pow(Crho, 2) * (rhoconst22*XtRes2 + rhoconst12*XtRes1 + rhoconst2Y*XtResY) - 1/Crho * (-XtRes1/(sigma1*sigma2) + rho1Y*XtResY/(sigma2*sigmaY));
  Ic(arma::span(a, a+a-1), a+a+b+6) = 2*(rho12*rho2Y - rho1Y)/pow(Crho, 2) * (rhoconst22*XtRes2 + rhoconst12*XtRes1 + rhoconst2Y*XtResY) - 1/Crho * (-2*rho1Y*XtRes2/sigma2square + rho2Y*XtRes1/(sigma1*sigma2) + rho12*XtResY/(sigma2*sigmaY));
  Ic(arma::span(a, a+a-1), a+a+b+7) = 2*(rho12*rho1Y - rho2Y)/pow(Crho, 2) * (rhoconst22*XtRes2 + rhoconst12*XtRes1 + rhoconst2Y*XtResY) - 1/Crho * (rho1Y*XtRes1/(sigma1*sigma2) - XtResY/(sigma2*sigmaY));
  
  //// beta
  //cout << "1-3" << endl;
  Ic(arma::span(a+a, a+a+b-1), arma::span(0, a-1)) = Ic(arma::span(0, a-1), arma::span(a+a, a+a+b-1)).t();
  Ic(arma::span(a+a, a+a+b-1), arma::span(a, a+a-1)) = Ic(arma::span(a, a+a-1), arma::span(a+a, a+a+b-1)).t();
  Ic(arma::span(a+a, a+a+b-1), arma::span(a+a, a+a+b-1)) = rhoconstYY / Crho * Z.t() * Z;
  Ic(arma::span(a+a, a+a+b-1), a+a+b) = rhoconstYY / Crho * Z.t() * E1;
  Ic(arma::span(a+a, a+a+b-1), a+a+b+1) = rhoconstYY / Crho * Z.t() * E2;
  Ic(arma::span(a+a, a+a+b-1), a+a+b+2) = 1 / (2*Crho) * rhoconst1Y / sigma1square * ZtRes1;
  Ic(arma::span(a+a, a+a+b-1), a+a+b+3) = 1 / (2*Crho) * rhoconst2Y / sigma2square * ZtRes2;
  Ic(arma::span(a+a, a+a+b-1), a+a+b+4) = 1 / (2*Crho) * (2*rhoconstYY/sigmaYsquare*ZtResY + rhoconst1Y/sigmaYsquare*ZtRes1 + rhoconst2Y/sigmaYsquare*ZtRes2);
  Ic(arma::span(a+a, a+a+b-1), a+a+b+5) = 2*(rho1Y*rho2Y - rho12)/pow(Crho, 2) * (rhoconstYY*ZtResY + rhoconst1Y*ZtRes1 + rhoconst2Y*ZtRes2) - 1/Crho * (-2*rho12*ZtResY/sigmaYsquare + rho2Y*ZtRes1/(sigma1*sigmaY) + rho1Y*ZtRes2/(sigma2*sigmaY));
  Ic(arma::span(a+a, a+a+b-1), a+a+b+6) = 2*(rho12*rho2Y - rho1Y)/pow(Crho, 2) * (rhoconstYY*ZtResY + rhoconst1Y*ZtRes1 + rhoconst2Y*ZtRes2) - 1/Crho * (-ZtRes1/(sigma1*sigmaY) + rho12*ZtRes2/(sigma2*sigmaY));
  Ic(arma::span(a+a, a+a+b-1), a+a+b+7) = 2*(rho12*rho1Y - rho2Y)/pow(Crho, 2) * (rhoconstYY*ZtResY + rhoconst1Y*ZtRes1 + rhoconst2Y*ZtRes2) - 1/Crho * (rho12*ZtRes1/(sigma1*sigmaY) - ZtRes2/(sigma2*sigmaY));
  
  //// gamma1
  //cout << "1-4" << endl;
  Ic(a+a+b, arma::span(0, a-1)) = Ic(arma::span(0, a-1), a+a+b).t();
  Ic(a+a+b, arma::span(a, a+a-1)) = Ic(arma::span(a, a+a-1), a+a+b).t();
  Ic(a+a+b, arma::span(a+a, a+a+b-1)) = Ic(arma::span(a+a, a+a+b-1), a+a+b).t();
  Ic(a+a+b, a+a+b) = rhoconstYY / Crho * sum(Ehat.col(2));
  Ic(a+a+b, a+a+b+1) = rhoconstYY / Crho * sum(Ehat.col(3));
  Ic(a+a+b, a+a+b+2) = 1 / (2*Crho) * rhoconst1Y / sigma1square * E1tRes1;
  Ic(a+a+b, a+a+b+3) = 1 / (2*Crho) * rhoconst2Y / sigma2square * E1tRes2;
  Ic(a+a+b, a+a+b+4) = 1 / (2*Crho) * (2*rhoconstYY/sigmaYsquare*E1tResY + rhoconst1Y/sigmaYsquare*E1tRes1 + rhoconst2Y/sigmaYsquare*E1tRes2);
  Ic(a+a+b, a+a+b+5) = 2*(rho1Y*rho2Y - rho12)/pow(Crho, 2) * (rhoconstYY*E1tResY + rhoconst1Y*E1tRes1 + rhoconst2Y*E1tRes2) - 1/Crho * (-2*rho12*E1tResY/sigmaYsquare + rho2Y*E1tRes1/(sigma1*sigmaY) + rho1Y*E1tRes2/(sigma2*sigmaY));
  Ic(a+a+b, a+a+b+6) = 2*(rho12*rho2Y - rho1Y)/pow(Crho, 2) * (rhoconstYY*E1tResY + rhoconst1Y*E1tRes1 + rhoconst2Y*E1tRes2) - 1/Crho * (-E1tRes1/(sigma1*sigmaY) + rho12*E1tRes2/(sigma2*sigmaY));
  Ic(a+a+b, a+a+b+7) = 2*(rho12*rho1Y - rho2Y)/pow(Crho, 2) * (rhoconstYY*E1tResY + rhoconst1Y*E1tRes1 + rhoconst2Y*E1tRes2) - 1/Crho * (rho12*E1tRes1/(sigma1*sigmaY) - E1tRes2/(sigma2*sigmaY));
  
  //// gamma2
  //cout << "1-5" << endl;
  Ic(a+a+b+1, arma::span(0, a-1)) = Ic(arma::span(0, a-1), a+a+b+1).t();
  Ic(a+a+b+1, arma::span(a, a+a-1)) = Ic(arma::span(a, a+a-1), a+a+b+1).t();
  Ic(a+a+b+1, arma::span(a+a, a+a+b-1)) = Ic(arma::span(a+a, a+a+b-1), a+a+b+1).t();
  Ic(a+a+b+1, a+a+b) = Ic(a+a+b, a+a+b+1);
  Ic(a+a+b+1, a+a+b+1) = rhoconstYY / Crho * sum(Ehat.col(4));
  Ic(a+a+b+1, a+a+b+2) = 1 / (2*Crho) * rhoconst1Y / sigma1square * E2tRes1;
  Ic(a+a+b+1, a+a+b+3) = 1 / (2*Crho) * rhoconst2Y / sigma2square * E2tRes2;
  Ic(a+a+b+1, a+a+b+4) = 1 / (2*Crho) * (2*rhoconstYY/sigmaYsquare*E2tResY + rhoconst1Y/sigmaYsquare*E2tRes1 + rhoconst2Y/sigmaYsquare*E2tRes2);
  Ic(a+a+b+1, a+a+b+5) = 2*(rho1Y*rho2Y - rho12)/pow(Crho, 2) * (rhoconstYY*E2tResY + rhoconst1Y*E2tRes1 + rhoconst2Y*E2tRes2) - 1/Crho * (-2*rho12*E2tResY/sigmaYsquare + rho2Y*E2tRes1/(sigma1*sigmaY) + rho1Y*E2tRes2/(sigma2*sigmaY));
  Ic(a+a+b+1, a+a+b+6) = 2*(rho12*rho2Y - rho1Y)/pow(Crho, 2) * (rhoconstYY*E2tResY + rhoconst1Y*E2tRes1 + rhoconst2Y*E2tRes2) - 1/Crho * (-E2tRes1/(sigma1*sigmaY) + rho12*E2tRes2/(sigma2*sigmaY));
  Ic(a+a+b+1, a+a+b+7) = 2*(rho12*rho1Y - rho2Y)/pow(Crho, 2) * (rhoconstYY*E2tResY + rhoconst1Y*E2tRes1 + rhoconst2Y*E2tRes2) - 1/Crho * (rho12*E2tRes1/(sigma1*sigmaY) - E2tRes2/(sigma2*sigmaY));
  
  //// sigma1square
  //cout << "1-6" << endl;
  Ic(a+a+b+2, arma::span(0, a-1)) = Ic(arma::span(0, a-1), a+a+b+2).t();
  Ic(a+a+b+2, arma::span(a, a+a-1)) = Ic(arma::span(a, a+a-1), a+a+b+2).t();
  Ic(a+a+b+2, arma::span(a+a, a+a+b-1)) = Ic(arma::span(a+a, a+a+b-1), a+a+b+2).t();
  Ic(a+a+b+2, a+a+b) = Ic(a+a+b, a+a+b+2);
  Ic(a+a+b+2, a+a+b+1) = Ic(a+a+b+1, a+a+b+2);
  Ic(a+a+b+2, a+a+b+2) = n / (4*sigma1square*sigma1square) * (1 + (1 - rho2Y*rho2Y) / Crho);
  Ic(a+a+b+2, a+a+b+3) = n * rho12 * (rho1Y*rho2Y - rho12) / (4*sigma1square*sigma2square*Crho);
  Ic(a+a+b+2, a+a+b+4) = n * rho1Y * (rho12*rho2Y - rho1Y) / (4*sigma1square*sigmaYsquare*Crho);
  Ic(a+a+b+2, a+a+b+5) = n * (rho1Y*rho2Y - rho12) / (2*sigma1square*Crho);
  Ic(a+a+b+2, a+a+b+6) = n * (rho12*rho2Y - rho1Y) / (2*sigma1square*Crho);
  Ic(a+a+b+2, a+a+b+7) = 0;
  
  //// sigma2square
  //cout << "1-7" << endl;
  Ic(a+a+b+3, arma::span(0, a-1)) = Ic(arma::span(0, a-1), a+a+b+3).t();
  Ic(a+a+b+3, arma::span(a, a+a-1)) = Ic(arma::span(a, a+a-1), a+a+b+3).t();
  Ic(a+a+b+3, arma::span(a+a, a+a+b-1)) = Ic(arma::span(a+a, a+a+b-1), a+a+b+3).t();
  Ic(a+a+b+3, a+a+b) = Ic(a+a+b, a+a+b+3);
  Ic(a+a+b+3, a+a+b+1) = Ic(a+a+b+1, a+a+b+3);
  Ic(a+a+b+3, a+a+b+2) = Ic(a+a+b+2, a+a+b+3);
  Ic(a+a+b+3, a+a+b+3) = n / (4*sigma2square*sigma2square) * (1 + (1 - rho1Y*rho1Y) / Crho);
  Ic(a+a+b+3, a+a+b+4) = n * rho2Y * (rho12*rho1Y - rho2Y) / (4*sigma2square*sigmaYsquare*Crho);
  Ic(a+a+b+3, a+a+b+5) = n * (rho1Y*rho2Y - rho12) / (2*sigma2square*Crho);
  Ic(a+a+b+3, a+a+b+6) = 0;
  Ic(a+a+b+3, a+a+b+7) = n * (rho12*rho1Y - rho2Y) / (2*sigma2square*Crho);
  
  //// sigmaYsquare
  //cout << "1-8" << endl;
  Ic(a+a+b+4, arma::span(0, a-1)) = Ic(arma::span(0, a-1), a+a+b+4).t();
  Ic(a+a+b+4, arma::span(a, a+a-1)) = Ic(arma::span(a, a+a-1), a+a+b+4).t();
  Ic(a+a+b+4, arma::span(a+a, a+a+b-1)) = Ic(arma::span(a+a, a+a+b-1), a+a+b+4).t();
  Ic(a+a+b+4, a+a+b) = Ic(a+a+b, a+a+b+4);
  Ic(a+a+b+4, a+a+b+1) = Ic(a+a+b+1, a+a+b+4);
  Ic(a+a+b+4, a+a+b+2) = Ic(a+a+b+2, a+a+b+4);
  Ic(a+a+b+4, a+a+b+3) = Ic(a+a+b+3, a+a+b+4);
  Ic(a+a+b+4, a+a+b+4) = n / (4*sigmaYsquare*sigmaYsquare) * (1 + (1 - rho12*rho12) / Crho);
  Ic(a+a+b+4, a+a+b+5) = 0;
  Ic(a+a+b+4, a+a+b+6) = n * (rho12*rho2Y - rho1Y) / (2*sigmaYsquare*Crho);
  Ic(a+a+b+4, a+a+b+7) = n * (rho12*rho1Y - rho2Y) / (2*sigmaYsquare*Crho);
  
  //// rho12
  //cout << "1-9" << endl;
  Ic(a+a+b+5, arma::span(0, a-1)) = Ic(arma::span(0, a-1), a+a+b+5).t();
  Ic(a+a+b+5, arma::span(a, a+a-1)) = Ic(arma::span(a, a+a-1), a+a+b+5).t();
  Ic(a+a+b+5, arma::span(a+a, a+a+b-1)) = Ic(arma::span(a+a, a+a+b-1), a+a+b+5).t();
  Ic(a+a+b+5, a+a+b) = Ic(a+a+b, a+a+b+5);
  Ic(a+a+b+5, a+a+b+1) = Ic(a+a+b+1, a+a+b+5);
  Ic(a+a+b+5, a+a+b+2) = Ic(a+a+b+2, a+a+b+5);
  Ic(a+a+b+5, a+a+b+3) = Ic(a+a+b+3, a+a+b+5);
  Ic(a+a+b+5, a+a+b+4) = Ic(a+a+b+4, a+a+b+5);
  Ic(a+a+b+5, a+a+b+5) = 2*n/pow(Crho, 2) * pow(rho1Y*rho2Y-rho12, 2) + n/Crho;
  Ic(a+a+b+5, a+a+b+6) = 2*n/pow(Crho, 2) * (rho1Y*rho2Y - rho12) * (rho12*rho2Y - rho1Y) - n*rho2Y/Crho;
  Ic(a+a+b+5, a+a+b+7) = 2*n/pow(Crho, 2) * (rho1Y*rho2Y - rho12) * (rho12*rho1Y - rho2Y) - n*rho1Y/Crho;
  
  //// rho1Y
  //cout << "1-10" << endl;
  Ic(a+a+b+6, arma::span(0, a-1)) = Ic(arma::span(0, a-1), a+a+b+6).t();
  Ic(a+a+b+6, arma::span(a, a+a-1)) = Ic(arma::span(a, a+a-1), a+a+b+6).t();
  Ic(a+a+b+6, arma::span(a+a, a+a+b-1)) = Ic(arma::span(a+a, a+a+b-1), a+a+b+6).t();
  Ic(a+a+b+6, a+a+b) = Ic(a+a+b, a+a+b+6);
  Ic(a+a+b+6, a+a+b+1) = Ic(a+a+b+1, a+a+b+6);
  Ic(a+a+b+6, a+a+b+2) = Ic(a+a+b+2, a+a+b+6);
  Ic(a+a+b+6, a+a+b+3) = Ic(a+a+b+3, a+a+b+6);
  Ic(a+a+b+6, a+a+b+4) = Ic(a+a+b+4, a+a+b+6);
  Ic(a+a+b+6, a+a+b+5) = Ic(a+a+b+5, a+a+b+6);
  Ic(a+a+b+6, a+a+b+6) = 2*n/pow(Crho, 2) * pow(rho12*rho2Y-rho1Y, 2) + n/Crho;
  Ic(a+a+b+6, a+a+b+7) = 2*n/pow(Crho, 2) * (rho12*rho2Y - rho1Y) * (rho12*rho1Y - rho2Y) - n*rho12/Crho;
  
  //// rho2Y
  //cout << "1-11" << endl;
  Ic(a+a+b+7, arma::span(0, a-1)) = Ic(arma::span(0, a-1), a+a+b+7).t();
  Ic(a+a+b+7, arma::span(a, a+a-1)) = Ic(arma::span(a, a+a-1), a+a+b+7).t();
  Ic(a+a+b+7, arma::span(a+a, a+a+b-1)) = Ic(arma::span(a+a, a+a+b-1), a+a+b+7).t();
  Ic(a+a+b+7, a+a+b) = Ic(a+a+b, a+a+b+7);
  Ic(a+a+b+7, a+a+b+1) = Ic(a+a+b+1, a+a+b+7);
  Ic(a+a+b+7, a+a+b+2) = Ic(a+a+b+2, a+a+b+7);
  Ic(a+a+b+7, a+a+b+3) = Ic(a+a+b+3, a+a+b+7);
  Ic(a+a+b+7, a+a+b+4) = Ic(a+a+b+4, a+a+b+7);
  Ic(a+a+b+7, a+a+b+5) = Ic(a+a+b+5, a+a+b+7);
  Ic(a+a+b+7, a+a+b+6) = Ic(a+a+b+6, a+a+b+7);
  Ic(a+a+b+7, a+a+b+7) = 2*n/pow(Crho, 2) * pow(rho12*rho1Y-rho2Y, 2) + n/Crho;
  
  return Ic;
}



// Louis Formula: Computation of E[I(mis|obs)]
arma::mat Info_mis_given_obs(arma::vec Y, arma::vec S1, arma::vec S2, arma::ivec R1, arma::ivec R2, arma::mat Z, arma::mat X,
                             arma::vec alpha1, arma::vec alpha2, arma::vec beta, double gamma1, double gamma2, arma::mat Sigma, arma::mat Ehat, 
                             arma::vec weight_w, arma::vec point_s){
  int n = Y.n_elem;
  double mu1i, mu2i, ES1i, ES2i, ES1i2, ES1iS2i, ES2i2;
  
  arma::mat Sigma_inv = Sigma.i();
  arma::mat Sigma_inv11 = Sigma_inv.submat(0, 0, 1, 1);
  arma::mat Sigma_inv12 = Sigma_inv.submat(0, 2, 1, 2);
  arma::mat Sigma_inv21 = Sigma_inv12.t();
  double Sigma_inv22 = Sigma_inv(2, 2);
  
  arma::vec gamma = arma::zeros(2);
  gamma(0) = gamma1;
  gamma(1) = gamma2;
  
  double sigma1square = Sigma(0, 0);
  double sigma1 = sqrt(sigma1square);
  double sigma2square = Sigma(1, 1);
  double sigma2 = sqrt(sigma2square);
  double sigmaYsquare = Sigma(2, 2);
  double sigmaY = sqrt(sigmaYsquare);
  double rho12 = Sigma(0, 1) / (sigma1 * sigma2);
  double rho1Y = Sigma(0, 2) / (sigma1 * sigmaY);
  double rho2Y = Sigma(1, 2) / (sigma2 * sigmaY);
  
  double Crho = 1 + 2*rho12*rho1Y*rho2Y - rho12*rho12 - rho1Y*rho1Y - rho2Y*rho2Y;
  double rhoconst11 = (1 - rho2Y*rho2Y) / sigma1square;
  double rhoconst22 = (1 - rho1Y*rho1Y) / sigma2square;
  double rhoconstYY = (1 - rho12*rho12) / sigmaYsquare;
  double rhoconst12 = (rho1Y*rho2Y - rho12) / (sigma1*sigma2);
  double rhoconst1Y = (rho12*rho2Y - rho1Y) / (sigma1*sigmaY);
  double rhoconst2Y = (rho12*rho1Y - rho2Y) / (sigma2*sigmaY);
  
  arma::mat alphaX_temp = arma::join_cols(alpha1.t() * X.t(), alpha2.t() * X.t());
  arma::mat A = (Sigma_inv11 - Sigma_inv12 * gamma.t() - gamma * Sigma_inv21 + gamma * Sigma_inv22 * gamma.t()).i();
  double a11 = A(0, 0);
  double a12 = A(0, 1);
  double a22 = A(1, 1);
  double a1giv2 = a11 - a12 * a12 / a22;
  double a2giv1 = a22 - a12 * a12 / a11;
  arma::mat B = (Sigma_inv11 - gamma * Sigma_inv21) * alphaX_temp - (Sigma_inv12 - gamma * Sigma_inv22) * (Y.t() - beta.t() * Z.t());
  arma::mat MU = A * B;
  
  //// variables needed for computations in different scenarios
  double mu2giv1i, mu1giv2i, L2giv1i, L1giv2i, L20i, L10i, U2giv1i, U1giv2i, U20i, U10i;
  //// number of quadrature points
  int K = point_s.n_elem;
  //// index for summation
  int index;
  //// other variables needed when both exposure variables are beyond detection limits
  double I00, I30, I21, I12, I03, I40, I31, I22, I13, I04;
  arma::vec xi1i, xi2i, tilde_mu1giv2i;
  
  int a = X.n_cols;
  int b = Z.n_cols;
  arma::mat temp = arma::zeros(a+a+b+8, a+a+b+8);
  arma::mat temp66 = arma::zeros(6, 6);
  arma::vec temp61 = arma::zeros(6);
  arma::mat Vi = arma::zeros(a+a+b+8, 6);
  
  double ealpha1 = 0;
  double ealpha2 = 0;
  double ebeta = 0;
  
  for(int i=0; i<n; i++){
    if(R1(i) != 0 || R2(i) != 0){
      temp61(0) = 1;
      temp61(1) = Ehat(i, 0);
      temp61(2) = Ehat(i, 1);
      temp61(3) = Ehat(i, 2);
      temp61(4) = Ehat(i, 3);
      temp61(5) = Ehat(i, 4);
      
      temp66(0, 0) = 1;
      temp66(0, 1) = Ehat(i, 0);
      temp66(0, 2) = Ehat(i, 1);
      temp66(0, 3) = Ehat(i, 2);
      temp66(0, 4) = Ehat(i, 3);
      temp66(0, 5) = Ehat(i, 4);
      temp66(1, 1) = Ehat(i, 2);
      temp66(1, 2) = Ehat(i, 3);
      temp66(2, 2) = Ehat(i, 4);
      temp66(1, 0) = temp66(0, 1);
      temp66(2, 0) = temp66(0, 2);
      temp66(3, 0) = temp66(0, 3);
      temp66(4, 0) = temp66(0, 4);
      temp66(5, 0) = temp66(0, 5);
      temp66(2, 1) = temp66(1, 2);
      
      mu1i = MU(0, i);
      mu2i = MU(1, i);
      if(R1(i) == 0){     // S1i is observed.
        ES1i = Ehat(i, 0);
        ES2i = Ehat(i, 1);
        ES1i2 = Ehat(i, 2);
        ES1iS2i = Ehat(i, 3);
        ES2i2 = Ehat(i, 4);
        
        if(R2(i) == 3){     // S2i is unmeasured.
          mu2giv1i = mu2i + (S1(i) - mu1i) * a12 / a11;
          I00 = 1;
          I30 = pow(S1(i), 3);
          I21 = pow(S1(i), 2) * ES2i;
          I12 = S1(i) * ES2i2;
          I03 = pow(mu2giv1i, 3) + 3*mu2giv1i*a2giv1;
          I40 = pow(S1(i), 4);
          I31 = I30 * ES2i;
          I22 = pow(S1(i), 2) * ES2i2;
          I13 = S1(i) * I03;
          I04 = pow(mu2giv1i, 4) + 6*pow(mu2giv1i, 2)*a2giv1 + 3*pow(a2giv1, 2);
        }
        else if(R2(i) == 1){    // S2i is below the lower detection limit.
          mu2giv1i = mu2i + (S1(i) - mu1i) * a12 / a11;
          L2giv1i = (S2(i) - mu2giv1i) / sqrt(a2giv1);
          I00 = 1;
          I30 = pow(S1(i), 3);
          I21 = pow(S1(i), 2) * ES2i;
          I12 = S1(i) * ES2i2;
          I03 = pow(mu2giv1i, 3) + 3*mu2giv1i*a2giv1 
            - (3*pow(mu2giv1i, 2)*sqrt(a2giv1) + 2*pow(a2giv1, 1.5) + 3*mu2giv1i*a2giv1*L2giv1i + pow(a2giv1, 1.5)*pow(L2giv1i, 2)) * arma::normpdf(L2giv1i) / arma::normcdf(L2giv1i);
          I40 = pow(S1(i), 4);
          I31 = I30 * ES2i;
          I22 = pow(S1(i), 2) * ES2i2;
          I13 = S1(i) * I03;
          I04 = pow(mu2giv1i, 4) + 6*pow(mu2giv1i, 2)*a2giv1 + 3*pow(a2giv1, 2) 
            - (4*pow(mu2giv1i, 3)*sqrt(a2giv1) + 8*mu2giv1i*pow(a2giv1, 1.5) + (6*pow(mu2giv1i, 2)*a2giv1 + 3*pow(a2giv1, 2)) * L2giv1i + 4*mu2giv1i*pow(a2giv1, 1.5)*pow(L2giv1i, 2) + pow(a2giv1, 2)*pow(L2giv1i, 3)) * arma::normpdf(L2giv1i) / arma::normcdf(L2giv1i);
        }
        else if(R2(i) == 2){    // S2i is above the upper detection limit.
          mu2giv1i = mu2i + (S1(i) - mu1i) * a12 / a11;
          U2giv1i = (-S2(i) + mu2giv1i) / sqrt(a2giv1);
          I00 = 1;
          I30 = pow(S1(i), 3);
          I21 = pow(S1(i), 2) * ES2i;
          I12 = S1(i) * ES2i2;
          I03 = pow(mu2giv1i, 3) + 3*mu2giv1i*a2giv1 
            + (3*pow(mu2giv1i, 2)*sqrt(a2giv1) + 2*pow(a2giv1, 1.5) - 3*mu2giv1i*a2giv1*U2giv1i + pow(a2giv1, 1.5)*pow(U2giv1i, 2)) * arma::normpdf(U2giv1i) / arma::normcdf(U2giv1i);
          I40 = pow(S1(i), 4);
          I31 = I30 * ES2i;
          I22 = pow(S1(i), 2) * ES2i2;
          I13 = S1(i) * I03;
          I04 = pow(mu2giv1i, 4) + 6*pow(mu2giv1i, 2)*a2giv1 + 3*pow(a2giv1, 2) 
            + (4*pow(mu2giv1i, 3)*sqrt(a2giv1) + 8*mu2giv1i*pow(a2giv1, 1.5) - (6*pow(mu2giv1i, 2)*a2giv1 + 3*pow(a2giv1, 2)) * U2giv1i + 4*mu2giv1i*pow(a2giv1, 1.5)*pow(U2giv1i, 2) - pow(a2giv1, 2)*pow(U2giv1i, 3)) * arma::normpdf(U2giv1i) / arma::normcdf(U2giv1i);
        }
        else{
          stop("Inadmissible values: elements of vector R2 should be chosen from (0, 1, 2, 3).");
        }
      }
      else if(R1(i) == 3){    // S1i is unmeasured.
        if(R2(i) == 0){   // S2i is observed.
          ES1i = Ehat(i, 0);
          ES2i = Ehat(i, 1);
          ES1i2 = Ehat(i, 2);
          ES1iS2i = Ehat(i, 3);
          ES2i2 = Ehat(i, 4);
          mu1giv2i = mu1i + (S2(i) - mu2i) * a12 / a22;
          
          I00 = 1;
          I30 = pow(mu1giv2i, 3) + 3*mu1giv2i*a1giv2;
          I21 = ES1i2 * S2(i);
          I12 = ES1i * pow(S2(i), 2);
          I03 = pow(S2(i), 3);
          I40 = pow(mu1giv2i, 4) + 6*pow(mu1giv2i, 2)*a1giv2 + 3*pow(a1giv2, 2);
          I31 = I30 * S2(i);
          I22 = ES1i2 * pow(S2(i), 2);
          I13 = ES1i * pow(S2(i), 3);
          I04 = pow(S2(i), 4);
        }
        else if(R2(i) == 3){
          I00 = 1;
          I30 = pow(mu1i, 3) + 3*mu1i*a11;
          I21 = mu1i*mu1i*mu2i + 2*mu1i*a12 + mu2i*a11;
          I12 = mu1i*mu2i*mu2i + mu1i*a22 + 2*mu2i*a12;
          I03 = pow(mu2i, 3) + 3*mu2i*a22;
          I40 = pow(mu1i, 4) + 6*mu1i*mu1i*a11 + 3*a11*a11;
          I31 = pow(mu1i, 3)*mu2i + 3*mu1i*mu1i*a12 + 3*mu1i*mu2i*a11 + 3*a11*a12;
          I22 = mu1i*mu1i*mu2i*mu2i + mu1i*mu1i*a22 + 4*mu1i*mu2i*a12 + mu2i*mu2i*a11 + a11*a22 + 2*a12*a12;
          I13 = mu1i*pow(mu2i, 3) + 3*mu1i*mu2i*a22 + 3*mu2i*mu2i*a12 + 3*a12*a22;
          I04 = pow(mu2i, 4) + 6*mu2i*mu2i*a22 + 3*a22*a22;
        }
        else if(R2(i) == 1){
          L20i = (S2(i) - mu2i) / sqrt(a22);
          I00 = 1;
          I30 = pow(mu1i, 3) + 3*mu1i*a11 
            - (3*mu1i*mu1i*a12/sqrt(a22) + 3*a11*a12/sqrt(a22) - pow(a12, 3)/pow(a22, 1.5) 
                 + 3*mu1i*pow(a12, 2)/a22*L20i + pow(a12, 3)/pow(a22, 1.5)*L20i*L20i) * arma::normpdf(L20i) / arma::normcdf(L20i);
          I21 = mu1i*mu1i*mu2i + 2*mu1i*a12 + mu2i*a11
            - (mu1i*mu1i*sqrt(a22) + 2*mu1i*mu2i*a12/sqrt(a22) + a11*sqrt(a22) + a12*a12/sqrt(a22) 
                 + (2*mu1i*a12 + mu2i*a12*a12/a22) * L20i + a12*a12/sqrt(a22)*L20i*L20i ) * arma::normpdf(L20i) / arma::normcdf(L20i);
          I12 = mu1i*mu2i*mu2i + mu1i*a22 + 2*mu2i*a12
            - (2*mu1i*mu2i*sqrt(a22) + mu2i*mu2i*a12/sqrt(a22) + 2*a12*sqrt(a22) 
                 + (mu1i*a22 + 2*mu2i*a12) * L20i + a12*sqrt(a22)*L20i*L20i ) * arma::normpdf(L20i) / arma::normcdf(L20i);
          I03 = pow(mu2i, 3) + 3*mu2i*a22 
            - (3*mu2i*mu2i*sqrt(a22) + 2*pow(a22, 1.5) + 3*mu2i*a22*L20i + pow(a22, 1.5)*L20i*L20i) * arma::normpdf(L20i) / arma::normcdf(L20i);
          I40 = pow(mu1i, 4) + 6*mu1i*mu1i*a11 + 3*a11*a11
            - (4*pow(mu1i, 3)*a12/sqrt(a22) + 12*mu1i*a11*a12/sqrt(a22) - 4*mu1i*pow(a12, 3)/pow(a22, 1.5) 
                 + (6*mu1i*mu1i*a12*a12/a22 + 6*a11*a12*a12/a22 - 3*pow(a12, 4)/pow(a22, 2)) * L20i 
                 + 4*mu1i*pow(a12, 3)/pow(a22, 1.5)*L20i*L20i + pow(a12, 4)/pow(a22, 2)*pow(L20i, 3) ) * arma::normpdf(L20i) / arma::normcdf(L20i);
          I31 = pow(mu1i, 3)*mu2i + 3*mu1i*mu1i*a12 + 3*mu1i*mu2i*a11 + 3*a11*a12
            - (pow(mu1i, 3)*sqrt(a22) + 3*mu1i*mu1i*mu2i*a12/sqrt(a22) + 3*mu1i*a11*sqrt(a22) + 3*mu1i*a12*a12/sqrt(a22) + 3*mu2i*a11*a12/sqrt(a22) - mu2i*pow(a12, 3)/pow(a22, 1.5)
                 + (3*mu1i*mu1i*a12 + 3*mu1i*mu2i*a12*a12/a22 + 3*a11*a12) * L20i 
                 + (3*mu1i*a12*a12/sqrt(a22) + mu2i*pow(a12, 3)/pow(a22, 1.5)) *L20i*L20i + pow(a12, 3)/a22*pow(L20i, 3) ) * arma::normpdf(L20i) / arma::normcdf(L20i);
          I22 = mu1i*mu1i*mu2i*mu2i + mu1i*mu1i*a22 + 4*mu1i*mu2i*a12 + mu2i*mu2i*a11 + a11*a22 + 2*a12*a12
            - (2*mu1i*mu1i*mu2i*sqrt(a22) + 2*mu1i*mu2i*mu2i*a12/sqrt(a22) + 4*mu1i*a12*sqrt(a22) + 2*mu2i*a11*sqrt(a22) + 2*mu2i*a12*a12/sqrt(a22)
                 + (mu1i*mu1i*a22 + 4*mu1i*mu2i*a12 + mu2i*mu2i*a12*a12/a22 + a11*a22 + 2*a12*a12 ) * L20i
                 + (2*mu1i*a12*sqrt(a22) + 2*mu2i*a12*a12/sqrt(a22)) *L20i*L20i + a12*a12*pow(L20i, 3) ) * arma::normpdf(L20i) / arma::normcdf(L20i);
          I13 = mu1i*pow(mu2i, 3) + 3*mu1i*mu2i*a22 + 3*mu2i*mu2i*a12 + 3*a12*a22
            - (3*mu1i*mu2i*mu2i*sqrt(a22) + pow(mu2i, 3)*a12/sqrt(a22) + 2*mu1i*pow(a22, 1.5) + 6*mu2i*a12*sqrt(a22)
                 + (3*mu1i*mu2i*a22 + 3*mu2i*mu2i*a12 + 3*a12*a22) * L20i
                 + (mu1i*pow(a22, 1.5) + 3*mu2i*a12*sqrt(a22)) *L20i*L20i + a12*a22*pow(L20i, 3) ) * arma::normpdf(L20i) / arma::normcdf(L20i);
          I04 = pow(mu2i, 4) + 6*mu2i*mu2i*a22 + 3*a22*a22
            - (4*pow(mu2i, 3)*sqrt(a22) + 8*mu2i*pow(a22, 1.5) + (6*mu2i*mu2i*a22 + 3*a22*a22) * L20i + 4*mu2i*pow(a22, 1.5)*L20i*L20i + a22*a22*pow(L20i, 3)) * arma::normpdf(L20i) / arma::normcdf(L20i);
        }
        else if(R2(i) == 2){
          U20i = (-S2(i) + mu2i) / sqrt(a22);
          I00 = 1;
          I30 = pow(mu1i, 3) + 3*mu1i*a11 
            + (3*mu1i*mu1i*a12/sqrt(a22) + 3*a11*a12/sqrt(a22) - pow(a12, 3)/pow(a22, 1.5) 
                 - 3*mu1i*pow(a12, 2)/a22*U20i + pow(a12, 3)/pow(a22, 1.5)*U20i*U20i) * arma::normpdf(U20i) / arma::normcdf(U20i);
          I21 = mu1i*mu1i*mu2i + 2*mu1i*a12 + mu2i*a11
            + (mu1i*mu1i*sqrt(a22) + 2*mu1i*mu2i*a12/sqrt(a22) + a11*sqrt(a22) + a12*a12/sqrt(a22) 
                 - (2*mu1i*a12 + mu2i*a12*a12/a22) * U20i + a12*a12/sqrt(a22)*U20i*U20i ) * arma::normpdf(U20i) / arma::normcdf(U20i);
          I12 = mu1i*mu2i*mu2i + mu1i*a22 + 2*mu2i*a12
            + (2*mu1i*mu2i*sqrt(a22) + mu2i*mu2i*a12/sqrt(a22) + 2*a12*sqrt(a22)
                 - (mu1i*a22 + 2*mu2i*a12) * U20i + a12*sqrt(a22)*U20i*U20i ) * arma::normpdf(U20i) / arma::normcdf(U20i);
          I03 = pow(mu2i, 3) + 3*mu2i*a22
            + (3*mu2i*mu2i*sqrt(a22) + 2*pow(a22, 1.5) - 3*mu2i*a22*U20i + pow(a22, 1.5)*U20i*U20i) * arma::normpdf(U20i) / arma::normcdf(U20i);
          I40 = pow(mu1i, 4) + 6*mu1i*mu1i*a11 + 3*a11*a11
            + (4*pow(mu1i, 3)*a12/sqrt(a22) + 12*mu1i*a11*a12/sqrt(a22) - 4*mu1i*pow(a12, 3)/pow(a22, 1.5)
                 - (6*mu1i*mu1i*a12*a12/a22 + 6*a11*a12*a12/a22 - 3*pow(a12, 4)/pow(a22, 2)) * U20i
                 + 4*mu1i*pow(a12, 3)/pow(a22, 1.5)*U20i*U20i - pow(a12, 4)/pow(a22, 2)*pow(U20i, 3) ) * arma::normpdf(U20i) / arma::normcdf(U20i);
          I31 = pow(mu1i, 3)*mu2i + 3*mu1i*mu1i*a12 + 3*mu1i*mu2i*a11 + 3*a11*a12
            + (pow(mu1i, 3)*sqrt(a22) + 3*mu1i*mu1i*mu2i*a12/sqrt(a22) + 3*mu1i*a11*sqrt(a22) + 3*mu1i*a12*a12/sqrt(a22) + 3*mu2i*a11*a12/sqrt(a22) - mu2i*pow(a12, 3)/pow(a22, 1.5)
                 - (3*mu1i*mu1i*a12 + 3*mu1i*mu2i*a12*a12/a22 + 3*a11*a12) * U20i
                 + (3*mu1i*a12*a12/sqrt(a22) + mu2i*pow(a12, 3)/pow(a22, 1.5)) *U20i*U20i - pow(a12, 3)/a22*pow(U20i, 3) ) * arma::normpdf(U20i) / arma::normcdf(U20i);
          I22 = mu1i*mu1i*mu2i*mu2i + mu1i*mu1i*a22 + 4*mu1i*mu2i*a12 + mu2i*mu2i*a11 + a11*a22 + 2*a12*a12
            + (2*mu1i*mu1i*mu2i*sqrt(a22) + 2*mu1i*mu2i*mu2i*a12/sqrt(a22) + 4*mu1i*a12*sqrt(a22) + 2*mu2i*a11*sqrt(a22) + 2*mu2i*a12*a12/sqrt(a22)
                 - (mu1i*mu1i*a22 + 4*mu1i*mu2i*a12 + mu2i*mu2i*a12*a12/a22 + a11*a22 + 2*a12*a12 ) * U20i
                 + (2*mu1i*a12*sqrt(a22) + 2*mu2i*a12*a12/sqrt(a22)) *U20i*U20i - a12*a12*pow(U20i, 3) ) * arma::normpdf(U20i) / arma::normcdf(U20i);
          I13 = mu1i*pow(mu2i, 3) + 3*mu1i*mu2i*a22 + 3*mu2i*mu2i*a12 + 3*a12*a22
            + (3*mu1i*mu2i*mu2i*sqrt(a22) + pow(mu2i, 3)*a12/sqrt(a22) + 2*mu1i*pow(a22, 1.5) + 6*mu2i*a12*sqrt(a22)
                 - (3*mu1i*mu2i*a22 + 3*mu2i*mu2i*a12 + 3*a12*a22) * U20i
                 + (mu1i*pow(a22, 1.5) + 3*mu2i*a12*sqrt(a22)) *U20i*U20i - a12*a22*pow(U20i, 3) ) * arma::normpdf(U20i) / arma::normcdf(U20i);
          I04 = pow(mu2i, 4) + 6*mu2i*mu2i*a22 + 3*a22*a22
            + (4*pow(mu2i, 3)*sqrt(a22) + 8*mu2i*pow(a22, 1.5) - (6*mu2i*mu2i*a22 + 3*a22*a22) * U20i + 4*mu2i*pow(a22, 1.5)*U20i*U20i - a22*a22*pow(U20i, 3)) * arma::normpdf(U20i) / arma::normcdf(U20i);
        }
        else{
          stop("Inadmissible values: elements of vector R2 should be chosen from (0, 1, 2, 3).");
        }
      }
      else if(R1(i) == 1){    // S1i is below the lower detection limit.
        if(R2(i) == 0){   // S2i is observed.
          ES1i = Ehat(i, 0);
          ES2i = Ehat(i, 1);
          ES1i2 = Ehat(i, 2);
          ES1iS2i = Ehat(i, 3);
          ES2i2 = Ehat(i, 4);
          mu1giv2i = mu1i + (S2(i) - mu2i) * a12 / a22;
          L1giv2i = (S1(i) - mu1giv2i) / sqrt(a1giv2);
          
          I00 = 1;
          I30 = pow(mu1giv2i, 3) + 3*mu1giv2i*a1giv2
            - (3*pow(mu1giv2i, 2)*sqrt(a1giv2) + 2*pow(a1giv2, 1.5) + 3*mu1giv2i*a1giv2*L1giv2i + pow(a1giv2, 1.5)*pow(L1giv2i, 2)) * arma::normpdf(L1giv2i) / arma::normcdf(L1giv2i);
          I21 = ES1i2 * S2(i);
          I12 = ES1i * pow(S2(i), 2);
          I03 = pow(S2(i), 3);
          I40 = pow(mu1giv2i, 4) + 6*pow(mu1giv2i, 2)*a1giv2 + 3*pow(a1giv2, 2)
            - (4*pow(mu1giv2i, 3)*sqrt(a1giv2) + 8*mu1giv2i*pow(a1giv2, 1.5) + (6*pow(mu1giv2i, 2)*a1giv2 + 3*pow(a1giv2, 2)) * L1giv2i + 4*mu1giv2i*pow(a1giv2, 1.5)*pow(L1giv2i, 2) + pow(a1giv2, 2)*pow(L1giv2i, 3)) * arma::normpdf(L1giv2i) / arma::normcdf(L1giv2i);
          I31 = I30 * S2(i);
          I22 = ES1i2 * pow(S2(i), 2);
          I13 = ES1i * pow(S2(i), 3);
          I04 = pow(S2(i), 4);
        }
        else if(R2(i) == 3){
          L10i = (S1(i) - mu1i) / sqrt(a11);
          I00 = 1;
          I30 = pow(mu1i, 3) + 3*mu1i*a11
            - (3*mu1i*mu1i*sqrt(a11) + 2*pow(a11, 1.5) + 3*mu1i*a11*L10i + pow(a11, 1.5)*L10i*L10i ) * arma::normpdf(L10i) / arma::normcdf(L10i);
          I21 = mu1i*mu1i*mu2i + 2*mu1i*a12 + mu2i*a11
            - (mu1i*mu1i*a12/sqrt(a11) + 2*mu1i*mu2i*sqrt(a11) + 2*sqrt(a11)*a12 
                 + (2*mu1i*a12 + mu2i*a11) * L10i + sqrt(a11)*a12*L10i*L10i ) * arma::normpdf(L10i) / arma::normcdf(L10i);
          I12 = mu1i*mu2i*mu2i + mu1i*a22 + 2*mu2i*a12
            - (2*mu1i*mu2i*a12/sqrt(a11) + mu2i*mu2i*sqrt(a11) + sqrt(a11)*a22 + a12*a12/sqrt(a11)
                 + (mu1i*a12*a12/a11 + 2*mu2i*a12) * L10i + a12*a12/sqrt(a11)*L10i*L10i ) * arma::normpdf(L10i) / arma::normcdf(L10i);
          I03 = pow(mu2i, 3) + 3*mu2i*a22
            - (3*mu2i*mu2i*a12/sqrt(a11) + 3*a12/sqrt(a11)*a22 - pow(a12, 3)/pow(a11, 1.5) 
                 + 3*mu2i*a12*a12/a11*L10i + pow(a12, 3)/pow(a11, 1.5)*L10i*L10i ) * arma::normpdf(L10i) / arma::normcdf(L10i);
          I40 = pow(mu1i, 4) + 6*mu1i*mu1i*a11 + 3*a11*a11
            - (4*pow(mu1i, 3)*sqrt(a11) + 8*mu1i*pow(a11, 1.5) + (6*mu1i*mu1i*a11 + 3*a11*a11) * L10i + 4*mu1i*pow(a11, 1.5)*L10i*L10i + a11*a11*pow(L10i, 3)) * arma::normpdf(L10i) / arma::normcdf(L10i);
          I31 = pow(mu1i, 3)*mu2i + 3*mu1i*mu1i*a12 + 3*mu1i*mu2i*a11 + 3*a11*a12
            - (pow(mu1i, 3)*a12/sqrt(a11) + 3*mu1i*mu1i*mu2i*sqrt(a11) + 6*mu1i*sqrt(a11)*a12 + 2*mu2i*pow(a11, 1.5)
                 + (3*mu1i*mu1i*a12 + 3*mu1i*mu2i*a11 + 3*a11*a12) * L10i 
                 + (3*mu1i*sqrt(a11)*a12 + mu2i*pow(a11, 1.5)) *L10i*L10i + a11*a12*pow(L10i, 3) ) * arma::normpdf(L10i) / arma::normcdf(L10i);
          I22 = mu1i*mu1i*mu2i*mu2i + mu1i*mu1i*a22 + 4*mu1i*mu2i*a12 + mu2i*mu2i*a11 + a11*a22 + 2*a12*a12
            - (2*mu1i*mu1i*mu2i*a12/sqrt(a11) + 2*mu1i*mu2i*mu2i*sqrt(a11) + 2*mu1i*sqrt(a11)*a22 + 2*mu1i*a12*a12/sqrt(a11) + 4*mu2i*sqrt(a11)*a12
                 + (mu1i*mu1i*a12*a12/a11 + 4*mu1i*mu2i*a12 + mu2i*mu2i*a11 + a11*a22 + 2*a12*a12) * L10i
                 + (2*mu1i*a12*a12/sqrt(a11) + 2*mu2i*sqrt(a11)*a12) *L10i*L10i + a12*a12*pow(L10i, 3) ) * arma::normpdf(L10i) / arma::normcdf(L10i);
          I13 = mu1i*pow(mu2i, 3) + 3*mu1i*mu2i*a22 + 3*mu2i*mu2i*a12 + 3*a12*a22
            - (3*mu1i*mu2i*mu2i*a12/sqrt(a11) + pow(mu2i, 3)*sqrt(a11) + 3*mu1i*a12/sqrt(a11)*a22 - mu1i*pow(a12, 3)/pow(a11, 1.5) + 3*mu2i*sqrt(a11)*a22 + 3*mu2i*a12*a12/sqrt(a11)
                 + (3*mu1i*mu2i*a12*a12/a11 + 3*mu2i*mu2i*a12 + 3*a12*a22) * L10i
                 + (mu1i*pow(a12, 3)/pow(a11, 1.5) + 3*mu2i*a12*a12/sqrt(a11)) *L10i*L10i + pow(a12, 3)/a11*pow(L10i, 3) ) * arma::normpdf(L10i) / arma::normcdf(L10i);
          I04 = pow(mu2i, 4) + 6*mu2i*mu2i*a22 + 3*a22*a22
            - (4*pow(mu2i, 3)*a12/sqrt(a11) + 12*mu2i*a12/sqrt(a11)*a22 - 4*mu2i*pow(a12, 3)/pow(a11, 1.5)
                 + (6*mu2i*mu2i*a12*a12/a11 + 6*a12*a12/a11*a22 - 3*pow(a12, 4)/pow(a11, 2)) * L10i
                 + 4*mu2i*pow(a12, 3)/pow(a11, 1.5)* L10i*L10i + pow(a12, 4)/pow(a11, 2)*pow(L10i, 3)) * arma::normpdf(L10i) / arma::normcdf(L10i);
        }
        else if(R2(i) == 1){
          // Preparations
          I00 = 0; I30 = 0; I21 = 0; I12 = 0; I03 = 0; I40 = 0; I31 = 0; I22 = 0; I13 = 0; I04 = 0;
          L20i = (S2(i) - mu2i) / sqrt(a22);
          tilde_mu1giv2i = mu1i * arma::ones(K) + sqrt(2)*a12/sqrt(a22) * point_s;
          xi1i = (S1(i) * arma::ones(K) - tilde_mu1giv2i) / sqrt(a1giv2);
          
          // Find the index that satisfies s[index - 1] <= L20i/sqrt(2) && s[index] > L20i/sqrt(2)
          index = find_largest_index(point_s, L20i/sqrt(2));
          
          // Computations of the sum
          for(int k=0; k<index; k++){
            I00 = I00 + weight_w(k) * arma::normcdf(xi1i(k));
            I30 = I30 + weight_w(k) * ((pow(tilde_mu1giv2i(k), 3) + 3*tilde_mu1giv2i(k)*a1giv2) * arma::normcdf(xi1i(k))
                                         - (3*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*sqrt(a1giv2) + 2*pow(a1giv2, 1.5) 
                                              + 3*tilde_mu1giv2i(k)*a1giv2*xi1i(k) + pow(a1giv2, 1.5)*xi1i(k)*xi1i(k)) * arma::normpdf(xi1i(k)) );
            I21 = I21 + weight_w(k) * (sqrt(2 * a22) * point_s(k) + mu2i) * ((tilde_mu1giv2i(k)*tilde_mu1giv2i(k) + a1giv2) * arma::normcdf(xi1i(k))
                                                                               - (2*tilde_mu1giv2i(k)*sqrt(a1giv2) + a1giv2*xi1i(k)) * arma::normpdf(xi1i(k)) );
            I12 = I12 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 2) * (tilde_mu1giv2i(k) * arma::normcdf(xi1i(k)) - sqrt(a1giv2) * arma::normpdf(xi1i(k)));
            I03 = I03 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 3) * arma::normcdf(xi1i(k));
            I40 = I40 + weight_w(k) * ((pow(tilde_mu1giv2i(k), 4) + 6*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*a1giv2 + 3*a1giv2*a1giv2) * arma::normcdf(xi1i(k))
                                         - (4*pow(tilde_mu1giv2i(k), 3)*sqrt(a1giv2) + 8*tilde_mu1giv2i(k)*pow(a1giv2, 1.5) 
                                              + (6*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*a1giv2 + 3*a1giv2*a1giv2) * xi1i(k) 
                                              + 4*tilde_mu1giv2i(k)*pow(a1giv2, 1.5)*xi1i(k)*xi1i(k) + a1giv2*a1giv2*pow(xi1i(k), 3) ) * arma::normpdf(xi1i(k)));
            I31 = I31 + weight_w(k) * (sqrt(2 * a22) * point_s(k) + mu2i) * ((pow(tilde_mu1giv2i(k), 3) + 3*tilde_mu1giv2i(k)*a1giv2) * arma::normcdf(xi1i(k))
                                                                               - (3*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*sqrt(a1giv2) + 2*pow(a1giv2, 1.5) 
                                                                                    + 3*tilde_mu1giv2i(k)*a1giv2*xi1i(k) + pow(a1giv2, 1.5)*xi1i(k)*xi1i(k)) * arma::normpdf(xi1i(k)) );
            I22 = I22 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 2) * ((tilde_mu1giv2i(k)*tilde_mu1giv2i(k) + a1giv2) * arma::normcdf(xi1i(k))
                                                                                     - (2*tilde_mu1giv2i(k)*sqrt(a1giv2) + a1giv2*xi1i(k)) * arma::normpdf(xi1i(k)) );
            I13 = I13 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 3) * (tilde_mu1giv2i(k) * arma::normcdf(xi1i(k)) - sqrt(a1giv2) * arma::normpdf(xi1i(k)));
            I04 = I04 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 4) * arma::normcdf(xi1i(k));
          }
        }
        else if(R2(i) == 2){
          // Preparations
          I00 = 0; I30 = 0; I21 = 0; I12 = 0; I03 = 0; I40 = 0; I31 = 0; I22 = 0; I13 = 0; I04 = 0;
          U20i = (-S2(i) + mu2i) / sqrt(a22);
          tilde_mu1giv2i = mu1i * arma::ones(K) + sqrt(2)*a12/sqrt(a22) * point_s;
          xi1i = (S1(i) * arma::ones(K) - tilde_mu1giv2i) / sqrt(a1giv2);
          
          // Find the index that satisfies s[index - 1] < -U20i/sqrt(2) && s[index] >= -U20i/sqrt(2)
          index = find_smallest_index(point_s, -U20i/sqrt(2));
          
          // Computations of the sum
          for(int k=index; k<K; k++){
            I00 = I00 + weight_w(k) * arma::normcdf(xi1i(k));
            I30 = I30 + weight_w(k) * ((pow(tilde_mu1giv2i(k), 3) + 3*tilde_mu1giv2i(k)*a1giv2) * arma::normcdf(xi1i(k))
                                         - (3*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*sqrt(a1giv2) + 2*pow(a1giv2, 1.5) 
                                              + 3*tilde_mu1giv2i(k)*a1giv2*xi1i(k) + pow(a1giv2, 1.5)*xi1i(k)*xi1i(k)) * arma::normpdf(xi1i(k)) );
            I21 = I21 + weight_w(k) * (sqrt(2 * a22) * point_s(k) + mu2i) * ((tilde_mu1giv2i(k)*tilde_mu1giv2i(k) + a1giv2) * arma::normcdf(xi1i(k))
                                                                               - (2*tilde_mu1giv2i(k)*sqrt(a1giv2) + a1giv2*xi1i(k)) * arma::normpdf(xi1i(k)) );
            I12 = I12 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 2) * (tilde_mu1giv2i(k) * arma::normcdf(xi1i(k)) - sqrt(a1giv2) * arma::normpdf(xi1i(k)));
            I03 = I03 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 3) * arma::normcdf(xi1i(k));
            I40 = I40 + weight_w(k) * ((pow(tilde_mu1giv2i(k), 4) + 6*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*a1giv2 + 3*a1giv2*a1giv2) * arma::normcdf(xi1i(k))
                                         - (4*pow(tilde_mu1giv2i(k), 3)*sqrt(a1giv2) + 8*tilde_mu1giv2i(k)*pow(a1giv2, 1.5)
                                              + (6*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*a1giv2 + 3*a1giv2*a1giv2) * xi1i(k)
                                              + 4*tilde_mu1giv2i(k)*pow(a1giv2, 1.5)*xi1i(k)*xi1i(k) + a1giv2*a1giv2*pow(xi1i(k), 3) ) * arma::normpdf(xi1i(k)));
            I31 = I31 + weight_w(k) * (sqrt(2 * a22) * point_s(k) + mu2i) * ((pow(tilde_mu1giv2i(k), 3) + 3*tilde_mu1giv2i(k)*a1giv2) * arma::normcdf(xi1i(k))
                                                                               - (3*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*sqrt(a1giv2) + 2*pow(a1giv2, 1.5)
                                                                                    + 3*tilde_mu1giv2i(k)*a1giv2*xi1i(k) + pow(a1giv2, 1.5)*xi1i(k)*xi1i(k)) * arma::normpdf(xi1i(k)) );
            I22 = I22 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 2) * ((tilde_mu1giv2i(k)*tilde_mu1giv2i(k) + a1giv2) * arma::normcdf(xi1i(k))
                                                                                     - (2*tilde_mu1giv2i(k)*sqrt(a1giv2) + a1giv2*xi1i(k)) * arma::normpdf(xi1i(k)) );
            I13 = I13 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 3) * (tilde_mu1giv2i(k) * arma::normcdf(xi1i(k)) - sqrt(a1giv2) * arma::normpdf(xi1i(k)));
            I04 = I04 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 4) * arma::normcdf(xi1i(k));
          }
        }
        else{
          stop("Inadmissible values: elements of vector R2 should be chosen from (0, 1, 2, 3).");
        }
      }
      else if(R1(i) == 2){    // S1i is above the upper detection limit.
        if(R2(i) == 0){   // S2i is observed.
          ES1i = Ehat(i, 0);
          ES2i = Ehat(i, 1);
          ES1i2 = Ehat(i, 2);
          ES1iS2i = Ehat(i, 3);
          ES2i2 = Ehat(i, 4);
          mu1giv2i = mu1i + (S2(i) - mu2i) * a12 / a22;
          U1giv2i = (-S1(i) + mu1giv2i) / sqrt(a1giv2);
          
          I00 = 1;
          I30 = pow(mu1giv2i, 3) + 3*mu1giv2i*a1giv2
            + (3*pow(mu1giv2i, 2)*sqrt(a1giv2) + 2*pow(a1giv2, 1.5) - 3*mu1giv2i*a1giv2*U1giv2i + pow(a1giv2, 1.5)*pow(U1giv2i, 2)) * arma::normpdf(U1giv2i) / arma::normcdf(U1giv2i);
          I21 = ES1i2 * S2(i);
          I12 = ES1i * pow(S2(i), 2);
          I03 = pow(S2(i), 3);
          I40 = pow(mu1giv2i, 4) + 6*pow(mu1giv2i, 2)*a1giv2 + 3*pow(a1giv2, 2)
            + (4*pow(mu1giv2i, 3)*sqrt(a1giv2) + 8*mu1giv2i*pow(a1giv2, 1.5) - (6*pow(mu1giv2i, 2)*a1giv2 + 3*pow(a1giv2, 2)) * U1giv2i + 4*mu1giv2i*pow(a1giv2, 1.5)*pow(U1giv2i, 2) - pow(a1giv2, 2)*pow(U1giv2i, 3)) * arma::normpdf(U1giv2i) / arma::normcdf(U1giv2i);
          I31 = I30 * S2(i);
          I22 = ES1i2 * pow(S2(i), 2);
          I13 = ES1i * pow(S2(i), 3);
          I04 = pow(S2(i), 4);
        }
        else if(R2(i) == 3){
          U10i = (-S1(i) + mu1i) / sqrt(a11);
          I00 = 1;
          I30 = pow(mu1i, 3) + 3*mu1i*a11
            + (3*mu1i*mu1i*sqrt(a11) + 2*pow(a11, 1.5) - 3*mu1i*a11*U10i + pow(a11, 1.5)*U10i*U10i ) * arma::normpdf(U10i) / arma::normcdf(U10i);
          I21 = mu1i*mu1i*mu2i + 2*mu1i*a12 + mu2i*a11
            + (mu1i*mu1i*a12/sqrt(a11) + 2*mu1i*mu2i*sqrt(a11) + 2*sqrt(a11)*a12 
                 - (2*mu1i*a12 + mu2i*a11) * U10i + sqrt(a11)*a12*U10i*U10i ) * arma::normpdf(U10i) / arma::normcdf(U10i);
          I12 = mu1i*mu2i*mu2i + mu1i*a22 + 2*mu2i*a12
            + (2*mu1i*mu2i*a12/sqrt(a11) + mu2i*mu2i*sqrt(a11) + sqrt(a11)*a22 + a12*a12/sqrt(a11)
                 - (mu1i*a12*a12/a11 + 2*mu2i*a12) * U10i + a12*a12/sqrt(a11)*U10i*U10i ) * arma::normpdf(U10i) / arma::normcdf(U10i);
          I03 = pow(mu2i, 3) + 3*mu2i*a22
            + (3*mu2i*mu2i*a12/sqrt(a11) + 3*a12/sqrt(a11)*a22 - pow(a12, 3)/pow(a11, 1.5)
                 - 3*mu2i*a12*a12/a11*U10i + pow(a12, 3)/pow(a11, 1.5)*U10i*U10i ) * arma::normpdf(U10i) / arma::normcdf(U10i);
          I40 = pow(mu1i, 4) + 6*mu1i*mu1i*a11 + 3*a11*a11
            + (4*pow(mu1i, 3)*sqrt(a11) + 8*mu1i*pow(a11, 1.5) - (6*mu1i*mu1i*a11 + 3*a11*a11) * U10i + 4*mu1i*pow(a11, 1.5)*U10i*U10i - a11*a11*pow(U10i, 3)) * arma::normpdf(U10i) / arma::normcdf(U10i);
          I31 = pow(mu1i, 3)*mu2i + 3*mu1i*mu1i*a12 + 3*mu1i*mu2i*a11 + 3*a11*a12
            + (pow(mu1i, 3)*a12/sqrt(a11) + 3*mu1i*mu1i*mu2i*sqrt(a11) + 6*mu1i*sqrt(a11)*a12 + 2*mu2i*pow(a11, 1.5)
                 - (3*mu1i*mu1i*a12 + 3*mu1i*mu2i*a11 + 3*a11*a12) * U10i
                 + (3*mu1i*sqrt(a11)*a12 + mu2i*pow(a11, 1.5)) *U10i*U10i - a11*a12*pow(U10i, 3) ) * arma::normpdf(U10i) / arma::normcdf(U10i);
          I22 = mu1i*mu1i*mu2i*mu2i + mu1i*mu1i*a22 + 4*mu1i*mu2i*a12 + mu2i*mu2i*a11 + a11*a22 + 2*a12*a12
            + (2*mu1i*mu1i*mu2i*a12/sqrt(a11) + 2*mu1i*mu2i*mu2i*sqrt(a11) + 2*mu1i*sqrt(a11)*a22 + 2*mu1i*a12*a12/sqrt(a11) + 4*mu2i*sqrt(a11)*a12
                 - (mu1i*mu1i*a12*a12/a11 + 4*mu1i*mu2i*a12 + mu2i*mu2i*a11 + a11*a22 + 2*a12*a12) * U10i
                 + (2*mu1i*a12*a12/sqrt(a11) + 2*mu2i*sqrt(a11)*a12) *U10i*U10i - a12*a12*pow(U10i, 3) ) * arma::normpdf(U10i) / arma::normcdf(U10i);
          I13 = mu1i*pow(mu2i, 3) + 3*mu1i*mu2i*a22 + 3*mu2i*mu2i*a12 + 3*a12*a22
            + (3*mu1i*mu2i*mu2i*a12/sqrt(a11) + pow(mu2i, 3)*sqrt(a11) + 3*mu1i*a12/sqrt(a11)*a22 - mu1i*pow(a12, 3)/pow(a11, 1.5) + 3*mu2i*sqrt(a11)*a22 + 3*mu2i*a12*a12/sqrt(a11)
                 - (3*mu1i*mu2i*a12*a12/a11 + 3*mu2i*mu2i*a12 + 3*a12*a22) * U10i
                 + (mu1i*pow(a12, 3)/pow(a11, 1.5) + 3*mu2i*a12*a12/sqrt(a11)) *U10i*U10i - pow(a12, 3)/a11*pow(U10i, 3) ) * arma::normpdf(U10i) / arma::normcdf(U10i);
          I04 = pow(mu2i, 4) + 6*mu2i*mu2i*a22 + 3*a22*a22
            + (4*pow(mu2i, 3)*a12/sqrt(a11) + 12*mu2i*a12/sqrt(a11)*a22 - 4*mu2i*pow(a12, 3)/pow(a11, 1.5)
                 - (6*mu2i*mu2i*a12*a12/a11 + 6*a12*a12/a11*a22 - 3*pow(a12, 4)/pow(a11, 2)) * U10i
                 + 4*mu2i*pow(a12, 3)/pow(a11, 1.5)* U10i*U10i - pow(a12, 4)/pow(a11, 2)*pow(U10i, 3)) * arma::normpdf(U10i) / arma::normcdf(U10i);
        }
        else if(R2(i) == 1){
          // Preparations
          I00 = 0; I30 = 0; I21 = 0; I12 = 0; I03 = 0; I40 = 0; I31 = 0; I22 = 0; I13 = 0; I04 = 0;
          L20i = (S2(i) - mu2i) / sqrt(a22);
          tilde_mu1giv2i = mu1i * arma::ones(K) + sqrt(2)*a12/sqrt(a22) * point_s;
          xi2i = (-S1(i) * arma::ones(K) + tilde_mu1giv2i) / sqrt(a1giv2);
          
          // Find the index that satisfies s[index - 1] <= L20i/sqrt(2) && s[index] > L20i/sqrt(2)
          index = find_largest_index(point_s, L20i/sqrt(2));
          
          // Computations of the sum
          for(int k=0; k<index; k++){
            I00 = I00 + weight_w(k) * arma::normcdf(xi2i(k));
            I30 = I30 + weight_w(k) * ((pow(tilde_mu1giv2i(k), 3) + 3*tilde_mu1giv2i(k)*a1giv2) * arma::normcdf(xi2i(k))
                                         + (3*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*sqrt(a1giv2) + 2*pow(a1giv2, 1.5) 
                                              - 3*tilde_mu1giv2i(k)*a1giv2*xi2i(k) + pow(a1giv2, 1.5)*xi2i(k)*xi2i(k)) * arma::normpdf(xi2i(k)));
            I21 = I21 + weight_w(k) * (sqrt(2 * a22) * point_s(k) + mu2i) * ((tilde_mu1giv2i(k)*tilde_mu1giv2i(k) + a1giv2) * arma::normcdf(xi2i(k))
                                                                               + (2*tilde_mu1giv2i(k)*sqrt(a1giv2) - a1giv2*xi2i(k)) * arma::normpdf(xi2i(k)) );
            I12 = I12 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 2) * (tilde_mu1giv2i(k) * arma::normcdf(xi2i(k)) + sqrt(a1giv2) * arma::normpdf(xi2i(k)));
            I03 = I03 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 3) * arma::normcdf(xi2i(k));
            I40 = I40 + weight_w(k) * ((pow(tilde_mu1giv2i(k), 4) + 6*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*a1giv2 + 3*a1giv2*a1giv2) * arma::normcdf(xi2i(k))
                                         + (4*pow(tilde_mu1giv2i(k), 3)*sqrt(a1giv2) + 8*tilde_mu1giv2i(k)*pow(a1giv2, 1.5) 
                                              - (6*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*a1giv2 + 3*a1giv2*a1giv2) * xi2i(k)
                                              + 4*tilde_mu1giv2i(k)*pow(a1giv2, 1.5)*xi2i(k)*xi2i(k) - a1giv2*a1giv2*pow(xi2i(k), 3)) * arma::normpdf(xi2i(k)) );
            I31 = I31 + weight_w(k) * (sqrt(2 * a22) * point_s(k) + mu2i) * ((pow(tilde_mu1giv2i(k), 3) + 3*tilde_mu1giv2i(k)*a1giv2) * arma::normcdf(xi2i(k))
                                                                               + (3*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*sqrt(a1giv2) + 2*pow(a1giv2, 1.5) 
                                                                                    - 3*tilde_mu1giv2i(k)*a1giv2*xi2i(k) + pow(a1giv2, 1.5)*xi2i(k)*xi2i(k)) * arma::normpdf(xi2i(k)));
            I22 = I22 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 2) * ((tilde_mu1giv2i(k)*tilde_mu1giv2i(k) + a1giv2) * arma::normcdf(xi2i(k))
                                                                                     + (2*tilde_mu1giv2i(k)*sqrt(a1giv2) - a1giv2*xi2i(k)) * arma::normpdf(xi2i(k)) );
            I13 = I13 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 3) * (tilde_mu1giv2i(k) * arma::normcdf(xi2i(k)) + sqrt(a1giv2) * arma::normpdf(xi2i(k)));
            I04 = I04 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 4) * arma::normcdf(xi2i(k));
          }
        }
        else if(R2(i) == 2){
          // Preparations
          I00 = 0; I30 = 0; I21 = 0; I12 = 0; I03 = 0; I40 = 0; I31 = 0; I22 = 0; I13 = 0; I04 = 0;
          U20i = (-S2(i) + mu2i) / sqrt(a22);
          tilde_mu1giv2i = mu1i * arma::ones(K) + sqrt(2)*a12/sqrt(a22) * point_s;
          xi2i = (-S1(i) * arma::ones(K) + tilde_mu1giv2i) / sqrt(a1giv2);
          
          // Find the index that satisfies s[index - 1] < -U20i/sqrt(2) && s[index] >= -U20i/sqrt(2)
          index = find_smallest_index(point_s, -U20i/sqrt(2));
          
          // Computations of the sum
          for(int k=index; k<K; k++){
            I00 = I00 + weight_w(k) * arma::normcdf(xi2i(k));
            I30 = I30 + weight_w(k) * ((pow(tilde_mu1giv2i(k), 3) + 3*tilde_mu1giv2i(k)*a1giv2) * arma::normcdf(xi2i(k))
                                         + (3*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*sqrt(a1giv2) + 2*pow(a1giv2, 1.5)
                                              - 3*tilde_mu1giv2i(k)*a1giv2*xi2i(k) + pow(a1giv2, 1.5)*xi2i(k)*xi2i(k)) * arma::normpdf(xi2i(k)));
            I21 = I21 + weight_w(k) * (sqrt(2 * a22) * point_s(k) + mu2i) * ((tilde_mu1giv2i(k)*tilde_mu1giv2i(k) + a1giv2) * arma::normcdf(xi2i(k))
                                                                               + (2*tilde_mu1giv2i(k)*sqrt(a1giv2) - a1giv2*xi2i(k)) * arma::normpdf(xi2i(k)) );
            I12 = I12 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 2) * (tilde_mu1giv2i(k) * arma::normcdf(xi2i(k)) + sqrt(a1giv2) * arma::normpdf(xi2i(k)));
            I03 = I03 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 3) * arma::normcdf(xi2i(k));
            I40 = I40 + weight_w(k) * ((pow(tilde_mu1giv2i(k), 4) + 6*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*a1giv2 + 3*a1giv2*a1giv2) * arma::normcdf(xi2i(k))
                                         + (4*pow(tilde_mu1giv2i(k), 3)*sqrt(a1giv2) + 8*tilde_mu1giv2i(k)*pow(a1giv2, 1.5)
                                              - (6*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*a1giv2 + 3*a1giv2*a1giv2) * xi2i(k)
                                              + 4*tilde_mu1giv2i(k)*pow(a1giv2, 1.5)*xi2i(k)*xi2i(k) - a1giv2*a1giv2*pow(xi2i(k), 3)) * arma::normpdf(xi2i(k)) );
            I31 = I31 + weight_w(k) * (sqrt(2 * a22) * point_s(k) + mu2i) * ((pow(tilde_mu1giv2i(k), 3) + 3*tilde_mu1giv2i(k)*a1giv2) * arma::normcdf(xi2i(k))
                                                                               + (3*tilde_mu1giv2i(k)*tilde_mu1giv2i(k)*sqrt(a1giv2) + 2*pow(a1giv2, 1.5)
                                                                                    - 3*tilde_mu1giv2i(k)*a1giv2*xi2i(k) + pow(a1giv2, 1.5)*xi2i(k)*xi2i(k)) * arma::normpdf(xi2i(k)));
            I22 = I22 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 2) * ((tilde_mu1giv2i(k)*tilde_mu1giv2i(k) + a1giv2) * arma::normcdf(xi2i(k))
                                                                                     + (2*tilde_mu1giv2i(k)*sqrt(a1giv2) - a1giv2*xi2i(k)) * arma::normpdf(xi2i(k)) );
            I13 = I13 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 3) * (tilde_mu1giv2i(k) * arma::normcdf(xi2i(k)) + sqrt(a1giv2) * arma::normpdf(xi2i(k)));
            I04 = I04 + weight_w(k) * pow(sqrt(2 * a22) * point_s(k) + mu2i, 4) * arma::normcdf(xi2i(k));
          }
        }
        else{
          stop("Inadmissible values: elements of vector R2 should be chosen from (0, 1, 2, 3).");
        }
      }
      else{
        stop("Inadmissible values: elements of vector R1 should be chosen from (0, 1, 2, 3).");
      }
      temp66(1, 3) = I30 / I00;
      temp66(1, 4) = I21 / I00;
      temp66(1, 5) = I12 / I00;
      temp66(2, 3) = I21 / I00;
      temp66(2, 4) = I12 / I00;
      temp66(2, 5) = I03 / I00;
      temp66(3, 3) = I40 / I00;
      temp66(3, 4) = I31 / I00;
      temp66(3, 5) = I22 / I00;
      temp66(4, 4) = I22 / I00;
      temp66(4, 5) = I13 / I00;
      temp66(5, 5) = I04 / I00;
      temp66(3, 1) = temp66(1, 3);
      temp66(4, 1) = temp66(1, 4);
      temp66(5, 1) = temp66(1, 5);
      temp66(3, 2) = temp66(2, 3);
      temp66(4, 2) = temp66(2, 4);
      temp66(5, 2) = temp66(2, 5);
      temp66(4, 3) = temp66(3, 4);
      temp66(5, 3) = temp66(3, 5);
      temp66(5, 4) = temp66(4, 5);
      
      
      //// Matrix Vi
      ealpha1 = arma::as_scalar(X.row(i) * alpha1);
      ealpha2 = arma::as_scalar(X.row(i) * alpha2);
      ebeta = Y(i) - arma::as_scalar(Z.row(i) * beta);
      
      ////// V1i
      //cout << "3-1" << endl;
      Vi(arma::span(0, a-1), 0) = 1/Crho * (-rhoconst11*ealpha1 - rhoconst12*ealpha2 + rhoconst1Y*ebeta) * X.row(i).t();
      Vi(arma::span(a, a+a-1), 0) = 1/Crho * (-rhoconst22*ealpha2 - rhoconst12*ealpha1 + rhoconst2Y*ebeta) * X.row(i).t();
      Vi(arma::span(a+a, a+a+b-1), 0) = 1/Crho * (rhoconstYY*ebeta - rhoconst1Y*ealpha1 - rhoconst2Y*ealpha2) * Z.row(i).t();
      //Vi(a+a+b, 0) = 0;
      //Vi(a+a+b+1, 0) = 0;
      Vi(a+a+b+2, 0) = -1/(2*sigma1square) + 1/(2*Crho) * (rhoconst11/sigma1square*ealpha1*ealpha1 + rhoconst12/sigma1square*ealpha1*ealpha2 - rhoconst1Y/sigma1square*ealpha1*ebeta);
      Vi(a+a+b+3, 0) = -1/(2*sigma2square) + 1/(2*Crho) * (rhoconst22/sigma2square*ealpha2*ealpha2 + rhoconst12/sigma2square*ealpha1*ealpha2 - rhoconst2Y/sigma2square*ealpha2*ebeta);
      Vi(a+a+b+4, 0) = -1/(2*sigmaYsquare) + 1/(2*Crho) * (rhoconstYY/sigmaYsquare*ebeta*ebeta - rhoconst1Y/sigmaYsquare*ealpha1*ebeta - rhoconst2Y/sigmaYsquare*ealpha2*ebeta);
      Vi(a+a+b+5, 0) = -(rho1Y*rho2Y - rho12)/Crho 
        + (rho1Y*rho2Y - rho12)/pow(Crho, 2) * (rhoconst11*ealpha1*ealpha1 + rhoconst22*ealpha2*ealpha2 + rhoconstYY*ebeta*ebeta + 2*rhoconst12*ealpha1*ealpha2 - 2*rhoconst1Y*ealpha1*ebeta - 2*rhoconst2Y*ealpha2*ebeta)
        + 1/Crho * (rho12*ebeta*ebeta/sigmaYsquare + ealpha1*ealpha2/(sigma1*sigma2) + rho2Y*ealpha1*ebeta/(sigma1*sigmaY) + rho1Y*ealpha2*ebeta/(sigma2*sigmaY));
      Vi(a+a+b+6, 0) = -(rho12*rho2Y - rho1Y)/Crho 
        + (rho12*rho2Y - rho1Y)/pow(Crho, 2) * (rhoconst11*ealpha1*ealpha1 + rhoconst22*ealpha2*ealpha2 + rhoconstYY*ebeta*ebeta + 2*rhoconst12*ealpha1*ealpha2 - 2*rhoconst1Y*ealpha1*ebeta - 2*rhoconst2Y*ealpha2*ebeta)
        + 1/Crho * (rho1Y*ealpha2*ealpha2/sigma2square - rho2Y*ealpha1*ealpha2/(sigma1*sigma2) - ealpha1*ebeta/(sigma1*sigmaY) + rho12*ealpha2*ebeta/(sigma2*sigmaY));
      Vi(a+a+b+7, 0) = -(rho12*rho1Y - rho2Y)/Crho
        + (rho12*rho1Y - rho2Y)/pow(Crho, 2) * (rhoconst11*ealpha1*ealpha1 + rhoconst22*ealpha2*ealpha2 + rhoconstYY*ebeta*ebeta + 2*rhoconst12*ealpha1*ealpha2 - 2*rhoconst1Y*ealpha1*ebeta - 2*rhoconst2Y*ealpha2*ebeta)
        + 1/Crho * (rho2Y*ealpha1*ealpha1/sigma1square - rho1Y*ealpha1*ealpha2/(sigma1*sigma2) + rho12*ealpha1*ebeta/(sigma1*sigmaY) - ealpha2*ebeta/(sigma2*sigmaY));
      
      ////// V2i
      //cout << "3-2" << endl;
      Vi(arma::span(0, a-1), 1) = 1/Crho * (rhoconst11 - gamma1*rhoconst1Y) * X.row(i).t();
      Vi(arma::span(a, a+a-1), 1) = 1/Crho * (rhoconst12 - gamma1*rhoconst2Y) * X.row(i).t();
      Vi(arma::span(a+a, a+a+b-1), 1) = 1/Crho * (rhoconst1Y - gamma1*rhoconstYY) * Z.row(i).t();
      Vi(a+a+b, 1) = 1/Crho * (rhoconstYY*ebeta - rhoconst1Y*ealpha1 - rhoconst2Y*ealpha2);
      //Vi(a+a+b+1, 1) = 0;
      Vi(a+a+b+2, 1) = 1/(2*Crho) * (-2*rhoconst11/sigma1square*ealpha1 - rhoconst12/sigma1square*ealpha2 + rhoconst1Y/sigma1square*(ebeta + gamma1*ealpha1));
      Vi(a+a+b+3, 1) = 1/(2*Crho) * (-rhoconst12/sigma2square + gamma1*rhoconst2Y/sigma2square) * ealpha2;
      Vi(a+a+b+4, 1) = 1/(2*Crho) * (-2*rhoconstYY/sigmaYsquare*gamma1*ebeta + rhoconst1Y/sigmaYsquare*(ebeta + gamma1*ealpha1) + rhoconst2Y/sigmaYsquare*gamma1*ealpha2);
      Vi(a+a+b+5, 1) = (rho1Y*rho2Y - rho12)/pow(Crho, 2) * (-2*rhoconst11*ealpha1 - 2*rhoconstYY*gamma1*ebeta - 2*rhoconst12*ealpha2 + 2*rhoconst1Y*(ebeta + gamma1*ealpha1) + 2*rhoconst2Y*gamma1*ealpha2)
        + 1/Crho * (-2*rho12*gamma1*ebeta/sigmaYsquare - ealpha2/(sigma1*sigma2) - rho2Y*(ebeta + gamma1*ealpha1)/(sigma1*sigmaY) - rho1Y*gamma1*ealpha2/(sigma2*sigmaY));
      Vi(a+a+b+6, 1) = (rho12*rho2Y - rho1Y)/pow(Crho, 2) * (-2*rhoconst11*ealpha1 - 2*rhoconstYY*gamma1*ebeta - 2*rhoconst12*ealpha2 + 2*rhoconst1Y*(ebeta + gamma1*ealpha1) + 2*rhoconst2Y*gamma1*ealpha2)
        + 1/Crho * (rho2Y*ealpha2/(sigma1*sigma2) + (ebeta + gamma1*ealpha1)/(sigma1*sigmaY) - rho12*gamma1*ealpha2/(sigma2*sigmaY));
      Vi(a+a+b+7, 1) = (rho12*rho1Y - rho2Y)/pow(Crho, 2) * (-2*rhoconst11*ealpha1 - 2*rhoconstYY*gamma1*ebeta - 2*rhoconst12*ealpha2 + 2*rhoconst1Y*(ebeta + gamma1*ealpha1) + 2*rhoconst2Y*gamma1*ealpha2)
        + 1/Crho * (-2*rho2Y*ealpha1/sigma1square + rho1Y*ealpha2/(sigma1*sigma2) - rho12*(ebeta + gamma1*ealpha1)/(sigma1*sigmaY) + gamma1*ealpha2/(sigma2*sigmaY));
      
      ////// V3i
      //cout << "3-3" << endl;
      Vi(arma::span(0, a-1), 2) = 1/Crho * (rhoconst12 - gamma2*rhoconst1Y) * X.row(i).t();
      Vi(arma::span(a, a+a-1), 2) = 1/Crho * (rhoconst22 - gamma2*rhoconst2Y) * X.row(i).t();
      Vi(arma::span(a+a, a+a+b-1), 2) = 1/Crho * (rhoconst2Y - gamma2*rhoconstYY) * Z.row(i).t();
      //Vi(a+a+b, 2) = 0;
      Vi(a+a+b+1, 2) = 1/Crho * (rhoconstYY*ebeta - rhoconst1Y*ealpha1 - rhoconst2Y*ealpha2);
      Vi(a+a+b+2, 2) = 1/(2*Crho) * (-rhoconst12/sigma1square + gamma2 * rhoconst1Y/sigma1square) * ealpha1;
      Vi(a+a+b+3, 2) = 1/(2*Crho) * (-2*rhoconst22/sigma2square*ealpha2 - rhoconst12/sigma2square*ealpha1 + rhoconst2Y/sigma2square*(ebeta + gamma2*ealpha2));
      Vi(a+a+b+4, 2) = 1/(2*Crho) * (-2*rhoconstYY/sigmaYsquare*gamma2*ebeta + rhoconst1Y/sigmaYsquare*gamma2*ealpha1 + rhoconst2Y/sigmaYsquare*(ebeta + gamma2*ealpha2));
      Vi(a+a+b+5, 2) = (rho1Y*rho2Y - rho12)/pow(Crho, 2) * (-2*rhoconst22*ealpha2 - 2*rhoconstYY*gamma2*ebeta - 2*rhoconst12*ealpha1 + 2*rhoconst1Y*gamma2*ealpha1 + 2*rhoconst2Y*(ebeta + gamma2*ealpha2))
        + 1/Crho * (-2*rho12*gamma2*ebeta/sigmaYsquare - ealpha1/(sigma1*sigma2) - rho2Y*gamma2*ealpha1/(sigma1*sigmaY) - rho1Y*(ebeta + gamma2*ealpha2)/(sigma2*sigmaY));
      Vi(a+a+b+6, 2) = (rho12*rho2Y - rho1Y)/pow(Crho, 2) * (-2*rhoconst22*ealpha2 - 2*rhoconstYY*gamma2*ebeta - 2*rhoconst12*ealpha1 + 2*rhoconst1Y*gamma2*ealpha1 + 2*rhoconst2Y*(ebeta + gamma2*ealpha2))
        + 1/Crho * (-2*rho1Y*ealpha2/sigma2square + rho2Y*ealpha1/(sigma1*sigma2) + gamma2*ealpha1/(sigma1*sigmaY) - rho12*(ebeta + gamma2*ealpha2)/(sigma2*sigmaY));
      Vi(a+a+b+7, 2) = (rho12*rho1Y - rho2Y)/pow(Crho, 2) * (-2*rhoconst22*ealpha2 - 2*rhoconstYY*gamma2*ebeta - 2*rhoconst12*ealpha1 + 2*rhoconst1Y*gamma2*ealpha1 + 2*rhoconst2Y*(ebeta + gamma2*ealpha2))
        + 1/Crho * (rho1Y*ealpha1/(sigma1*sigma2) - rho12*gamma2*ealpha1/(sigma1*sigmaY) + (ebeta + gamma2*ealpha2)/(sigma2*sigmaY));
      
      ////// V4i
      //cout << "3-4" << endl;
      //Vi(arma::span(0, a-1), 3) = 0;
      //Vi(arma::span(a, a+a-1), 3) = 0;
      //Vi(arma::span(a+a, a+a+b-1), 3) = 0;
      Vi(a+a+b, 3) = 1/Crho * (rhoconst1Y - gamma1*rhoconstYY);
      //Vi(a+a+b+1, 3) = 0;
      Vi(a+a+b+2, 3) = 1/(2*Crho) * (rhoconst11/sigma1square - gamma1*rhoconst1Y/sigma1square);
      //Vi(a+a+b+3, 3) = 0;
      Vi(a+a+b+4, 3) = 1/(2*Crho) * (rhoconstYY/sigmaYsquare*gamma1*gamma1 - gamma1*rhoconst1Y/sigmaYsquare);
      Vi(a+a+b+5, 3) = (rho1Y*rho2Y - rho12)/pow(Crho, 2) * (rhoconst11 + gamma1*gamma1*rhoconstYY - 2*gamma1*rhoconst1Y) + 1/Crho * (rho12*gamma1*gamma1/sigmaYsquare + rho2Y*gamma1/(sigma1*sigmaY));
      Vi(a+a+b+6, 3) = (rho12*rho2Y - rho1Y)/pow(Crho, 2) * (rhoconst11 + gamma1*gamma1*rhoconstYY - 2*gamma1*rhoconst1Y) - 1/Crho * gamma1/(sigma1*sigmaY);
      Vi(a+a+b+7, 3) = (rho12*rho1Y - rho2Y)/pow(Crho, 2) * (rhoconst11 + gamma1*gamma1*rhoconstYY - 2*gamma1*rhoconst1Y) + 1/Crho * (rho2Y/sigma1square + gamma1*rho12/(sigma1*sigmaY));
      
      ////// V5i
      //cout << "3-5" << endl;
      //Vi(arma::span(0, a-1), 4) = 0;
      //Vi(arma::span(a, a+a-1), 4) = 0;
      //Vi(arma::span(a+a, a+a+b-1), 4) = 0;
      Vi(a+a+b, 4) = 1/Crho * (rhoconst2Y - gamma2*rhoconstYY);
      Vi(a+a+b+1, 4) = 1/Crho * (rhoconst1Y - gamma1*rhoconstYY);
      Vi(a+a+b+2, 4) = 1/(2*Crho) * (rhoconst12/sigma1square - gamma2*rhoconst1Y/sigma1square);
      Vi(a+a+b+3, 4) = 1/(2*Crho) * (rhoconst12/sigma2square - gamma1*rhoconst2Y/sigma2square);
      Vi(a+a+b+4, 4) = 1/(2*Crho) * (rhoconstYY/sigmaYsquare*2*gamma1*gamma2 - gamma2*rhoconst1Y/sigmaYsquare - gamma1*rhoconst2Y/sigmaYsquare);
      Vi(a+a+b+5, 4) = (rho1Y*rho2Y - rho12)/pow(Crho, 2) * (2*rhoconstYY*gamma1*gamma2 + 2*rhoconst12 - 2*gamma2*rhoconst1Y - 2*gamma1*rhoconst2Y)
        + 1/Crho * (rho12/sigmaYsquare*2*gamma1*gamma2 + 1/(sigma1*sigma2) + rho2Y*gamma2/(sigma1*sigmaY) + rho1Y*gamma1/(sigma2*sigmaY));
      Vi(a+a+b+6, 4) = (rho12*rho2Y - rho1Y)/pow(Crho, 2) * (2*rhoconstYY*gamma1*gamma2 + 2*rhoconst12 - 2*gamma2*rhoconst1Y - 2*gamma1*rhoconst2Y)
        + 1/Crho * (-rho2Y/(sigma1*sigma2) - gamma2/(sigma1*sigmaY) + rho12*gamma1/(sigma2*sigmaY));
      Vi(a+a+b+7, 4) = (rho12*rho1Y - rho2Y)/pow(Crho, 2) * (2*rhoconstYY*gamma1*gamma2 + 2*rhoconst12 - 2*gamma2*rhoconst1Y - 2*gamma1*rhoconst2Y)
        + 1/Crho * (-rho1Y/(sigma1*sigma2) + rho12*gamma2/(sigma1*sigmaY) - gamma1/(sigma2*sigmaY));
      
      ////// V6i
      //cout << "3-6" << endl;
      //Vi(arma::span(0, a-1), 5) = 0;
      //Vi(arma::span(a, a+a-1), 5) = 0;
      //Vi(arma::span(a+a, a+a+b-1), 5) = 0;
      //Vi(a+a+b, 5) = 0;
      Vi(a+a+b+1, 5) = 1/Crho * (rhoconst2Y - gamma2*rhoconstYY);
      //Vi(a+a+b+2, 5) = 0;
      Vi(a+a+b+3, 5) = 1/(2*Crho) * (rhoconst22/sigma2square - gamma2*rhoconst2Y/sigma2square);
      Vi(a+a+b+4, 5) = 1/(2*Crho) * (rhoconstYY/sigmaYsquare*gamma2*gamma2 - gamma2*rhoconst2Y/sigmaYsquare);
      Vi(a+a+b+5, 5) = (rho1Y*rho2Y - rho12)/pow(Crho, 2) * (rhoconst22 + gamma2*gamma2*rhoconstYY - 2*gamma2*rhoconst2Y) + 1/Crho * (rho12*gamma2*gamma2/sigmaYsquare + rho1Y*gamma2/(sigma2*sigmaY));
      Vi(a+a+b+6, 5) = (rho12*rho2Y - rho1Y)/pow(Crho, 2) * (rhoconst22 + gamma2*gamma2*rhoconstYY - 2*gamma2*rhoconst2Y) + 1/Crho * (rho1Y/sigma2square + gamma2*rho12/(sigma2*sigmaY));
      Vi(a+a+b+7, 5) = (rho12*rho1Y - rho2Y)/pow(Crho, 2) * (rhoconst22 + gamma2*gamma2*rhoconstYY - 2*gamma2*rhoconst2Y) - 1/Crho * gamma2/(sigma2*sigmaY);
      
      temp = temp + Vi * (temp66 - temp61 * temp61.t()) * Vi.t();
    }
  }
  
  return(temp);
}



// Computation of estimated covariance matrix
arma::mat est_cov(arma::vec Y, arma::vec S1, arma::vec S2, arma::ivec R1, arma::ivec R2, arma::mat Z, arma::mat X,
                  arma::vec alpha1, arma::vec alpha2, arma::vec beta, double gamma1, double gamma2, arma::mat Sigma, arma::mat Ehat, arma::vec weight_w, arma::vec point_s){
  arma::mat mat1 = Info_complete(X, Y, Z, Ehat, alpha1, alpha2, beta, gamma1, gamma2, Sigma);
  arma::mat mat2 = Info_mis_given_obs(Y, S1, S2, R1, R2, Z, X, alpha1, alpha2, beta, gamma1, gamma2, Sigma, Ehat, weight_w, point_s);
  return (mat1 - mat2).i();
}



// Implementation
struct output{
  arma::vec alpha1;
  arma::vec alpha2;
  arma::vec beta;
  double gamma1;
  double gamma2;
  arma::mat Sigma;
  arma::mat cov;
  int num_iter;
};



struct output MVMR_EM(arma::vec Y, arma::vec S1, arma::vec S2, arma::ivec R1, arma::ivec R2, arma::mat Z, arma::mat X,
                      arma::vec alpha1_init, arma::vec alpha2_init, arma::vec beta_init, double gamma1_init, double gamma2_init, arma::mat Sigma_init,
                      double epsilon, arma::vec QuaWeight, arma::vec QuaPoint){
  struct output mle;
  struct para para_current, para_update;
  int n = Y.n_elem;
  
  //cout << "1-1" << endl;
  arma::vec gamma_init = arma::zeros(2);
  gamma_init(0) = gamma1_init;
  gamma_init(1) = gamma2_init;
  
  para_current.alpha1 = alpha1_init;
  para_current.alpha2 = alpha2_init;
  para_current.beta = beta_init;
  para_current.gamma1 = gamma1_init;
  para_current.gamma2 = gamma2_init;
  para_current.longpara = arma::join_cols(arma::join_cols(alpha1_init, alpha2_init), arma::join_cols(beta_init, gamma_init));
  
  para_current.Sigma = Sigma_init;
  para_current.m = 0;
  
  //cout << "2" << endl;
  arma::mat Ehat = arma::zeros(n, 5);
  para_update = para_current;
  arma::mat est_cov_matrix = arma::zeros(para_current.longpara.n_elem + 6, para_current.longpara.n_elem + 6);
  
  //cout << "3" << endl;
  int count = 0;
  double dist = epsilon + 1;
  
  while((dist > epsilon) && (count <= 1000)){
    count = count + 1;
    para_current = para_update;
    
    //// E-step
    //cout << "3-1" << endl;
    Ehat = CondExpect_update(para_current.alpha1, para_current.alpha2, para_current.beta, para_current.gamma1, para_current.gamma2, para_current.Sigma,
                             Y, S1, S2, R1, R2, Z, X, QuaWeight, QuaPoint);
    
    //// M-step
    //cout << "3-2" << endl;
    para_update = RegPara_Varcomp_update(X, Y, Z, Ehat, para_current.alpha1, para_current.alpha2, para_current.beta, para_current.gamma1, para_current.gamma2, epsilon);
    
    //// distance update
    //cout << "3-3" << endl;
    dist = sqrt(sum(square(para_update.longpara - para_current.longpara)));
  }
  //cout << "count = " << count << endl;
  
  //// Computation of the estimated covariance matrix
  Ehat = CondExpect_update(para_update.alpha1, para_update.alpha2, para_update.beta, para_update.gamma1, para_update.gamma2, para_update.Sigma,
                           Y, S1, S2, R1, R2, Z, X, QuaWeight, QuaPoint);
  est_cov_matrix = est_cov(Y, S1, S2, R1, R2, Z, X,
                           para_update.alpha1, para_update.alpha2, para_update.beta, para_update.gamma1, para_update.gamma2, para_update.Sigma, Ehat, QuaWeight, QuaPoint);
  
  //// output
  mle.alpha1 = para_update.alpha1;
  mle.alpha2 = para_update.alpha2;
  mle.beta = para_update.beta;
  mle.gamma1 = para_update.gamma1;
  mle.gamma2 = para_update.gamma2;
  mle.Sigma = para_update.Sigma;
  mle.cov = est_cov_matrix;
  mle.num_iter = count;
  //cout << "5" << endl;
  
  return mle;
}



// [[Rcpp::export]]
List MVMR_estimation_EM(arma::vec Y, arma::vec S1, arma::vec S2, arma::ivec R1, arma::ivec R2, arma::mat Z, arma::mat X,
                        arma::vec alpha1_init, arma::vec alpha2_init, arma::vec beta_init, double gamma1_init, double gamma2_init, arma::mat Sigma_init,
                        double epsilon, arma::vec QuaWeight, arma::vec QuaPoint){
  //cout << "start" << endl;
  struct output mle = MVMR_EM(Y, S1, S2, R1, R2, Z, X, alpha1_init, alpha2_init, beta_init, gamma1_init, gamma2_init, Sigma_init, epsilon, QuaWeight, QuaPoint);
  //cout << "end" << endl;
  return List::create(mle.alpha1, mle.alpha2, mle.beta, mle.gamma1, mle.gamma2, mle.Sigma, mle.cov, mle.num_iter);
}

