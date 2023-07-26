// MIT License

// Copyright (c) 2018 Jonathan Ish-Horowicz

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// ----------------------------------------------------------------------------------
//  Implementation of the B-spline Mutual information estimation method
// ----------------------------------------------------------------------------------

#include "fastGeneMI.h"

using namespace Rcpp;

// ----------------------------------------------------------------------------------
//  Class that returns bspline coefficients
// ----------------------------------------------------------------------------------

// Constructor
Bspline::Bspline(const int k_, const int n_bins_, const arma::vec& x_, const arma::vec& t_)
{
  k = k_;
  n_bins = n_bins_;
  x = x_;
  n_pts = x.n_elem;
  t = t_;
  tmax = t.max();
  
  // Transform data to range of knot vector and trim maximum values
  z = ((x-x.min())/(x.max()-x.min())) * tmax;
  z.replace(tmax, tmax-1e-12);
}

void Bspline::print() const
{
  Rcout << "k=" << k << "\t #bins=" << n_bins << "\t#pts" << n_pts << std::endl;
  Rcout << "t^T=" << arma::trans(t) << std::endl;
  Rcout << "x^T=" << arma::trans(x) << std::endl;
  Rcout << "z^T=" << arma::trans(z) << std::endl;
}


// Return zero-th order basis function
arma::vec Bspline::basis0(const double zi) const
{
  arma::vec res(n_bins+k-1, arma::fill::zeros);
  int loc = arma::find(zi >= t).max();
  res(loc) = 1.0;
  return res;
}

arma::vec Bspline::basis(const double zi, const int p) const
{
  //Rcout << "Calling basis with p=" << p << " at point zi=" << zi << std::endl;
  arma::vec basis_p_minus_1;
  // 0-th order basis function
  if(p==0)
    return basis0(zi);
  
  // Higher orders
  else
    basis_p_minus_1 = basis(zi, p-1);
  
  arma::vec first_term_num = zi - t.rows(0, t.n_elem-p-1);
  arma::vec first_term_denom = t.rows(p, t.n_elem-1) - t.rows(0, t.n_elem-p-1);
  
  arma::vec second_term_num = t.rows(p+1, t.n_elem-1) - zi;
  arma::vec second_term_denom = t.rows(p+1, t.n_elem-1) - t.rows(1, t.n_elem-p-1);
  
  arma::vec first_term = first_term_num/first_term_denom;
  arma::vec second_term = second_term_num/second_term_denom;
  
  // Set inf to zero and remove nan
  first_term.elem(arma::find(arma::abs(first_term)==arma::datum::inf)).zeros();
  second_term.elem(arma::find(arma::abs(second_term)==arma::datum::inf)).zeros();
  first_term.replace(arma::datum::nan, 0.0);
  second_term.replace(arma::datum::nan, 0.0);
  
  return first_term.rows(0, first_term.n_elem-2) %
  basis_p_minus_1.rows(0, basis_p_minus_1.n_elem-2) +
  second_term % basis_p_minus_1.rows(1, basis_p_minus_1.n_elem-1);
}

// Evaluate the B-spline basis functions at each point and return
// the result as a sparse matrix
inline arma::sp_mat Bspline::compute_bspl_coeffs()
{
  // int ct(0);
  arma::mat res = arma::mat(n_pts, n_bins, arma::fill::zeros);
  for(int i(0); i<n_pts; ++i)
  {
    res.row(i) = basis(z[i], k-1).t();
  } 

  // Zero small values
  res.elem(arma::find(res<1e-10)).zeros();
  return arma::sp_mat(res);
  
}

// Wrapper for computing bspline coefficients
arma::sp_mat get_bspl_coeffs(const arma::vec& x, const int n_bins, const int k, const arma::vec& t)
{
  Bspline bspl = Bspline(k, n_bins, x, t);
  //bspl.print();
  return bspl.compute_bspl_coeffs();
}

// Get the knot vector
arma::vec get_knots(const int k, const int n_bins)
{
  int n_knots(k + n_bins);
  arma::vec t(n_knots, arma::fill::zeros);
  
  // Intermediate (non-zero) knots
  for(int i(k); i<n_bins; ++i)
  {
    t(i) = (double)i - (double)k + 1.0;
  }
  
  // End values
  t.rows(n_knots-k, n_knots-1) = ((double)n_bins - (double)k + 1.0) * arma::vec(k, arma::fill::ones);
  return t;
}


// Compute the mutual information matrix of expression data using the B-spline method using 
// splines of order k
// [[Rcpp::export]]
arma::mat mim_bspline_cpp(const arma::mat& expr_data, const int k, const int n_bins, const int n_cores)
{
  const int n_genes(expr_data.n_cols), n_samples(expr_data.n_rows);
  const int n_pairs = get_n_gene_pairs(n_genes);
  const arma::vec t = get_knots(k, n_bins);
  
  // Compute bspline coefficients in parallel
  int j;
  
  // Number of cores to use
  //const int n_cores = sysconf(_SC_NPROCESSORS_ONLN);
  omp_set_num_threads(n_cores);
  
  const arma::mat data = expr_data;
  std::vector<arma::sp_mat> bspl_coeffs(n_genes);
  #pragma omp parallel for shared(bspl_coeffs, n_genes, data, t, k, n_bins) schedule(auto) default(none)
  for(int j=0; j<n_genes; ++j)
  {
    bspl_coeffs[j] = get_bspl_coeffs(data.col(j), n_bins, k, t);
  }
  
  //Rcout << "Got bspline coefficients\n";
  
  // Compute marginal probabilities and entropies
  std::vector<double> h_marginals(n_genes);
  for(int j(0); j<n_genes; ++j)
  {
    arma::vec p_marginal = arma::vec(arma::trans(arma::sum(bspl_coeffs[j], 0)/(double)n_samples));
    h_marginals[j] = -arma::sum(p_marginal % arma::log(p_marginal + 1e-16));
  }
  
  //Rcout << "Got marginal probabilities and entropies\n";
  
  // Compute joint probabilities and entropies in parallel
  std::vector<double> h_joints(n_pairs);

  const std::vector<std::pair <int,int> > ij_pairs = get_ij_list(n_genes);

  #pragma omp parallel for shared(bspl_coeffs, n_pairs, n_samples, h_joints)
  for(int ij=0; ij<n_pairs; ++ij)
  {
    std::pair <int,int> ij_pair = ij_pairs[ij];
    int i(ij_pair.first), j(ij_pair.second);
    arma::mat p_joint = arma::mat((bspl_coeffs[i].t() * bspl_coeffs[j])/(double)n_samples);
    h_joints[ij] = -arma::accu(p_joint % arma::log(p_joint + 1e-16));
  }
  
  //Rcout << "Got joint probabilities and entropies\n";
  
  // Compute mutual information matrix
  arma::mat mim = arma::mat(n_genes, n_genes, arma::fill::zeros);
  int ij(0);
  for(int i(0); i<n_genes; ++i)
  {
    for(int j(i); j<n_genes; ++j)
    {
      mim(i,j) = h_marginals[i] + h_marginals[j] - h_joints[ij];
      mim(j,i) = mim(i,j);
      ++ij;
    }
  }
  
  return mim;
  
}

