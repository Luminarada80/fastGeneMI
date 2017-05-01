// ----------------------------------------------------------------------------------
//  C++ port of the (basic) Bayesian Blocks algorithm
// ----------------------------------------------------------------------------------

#include "fastGeneMI.h"

using namespace Rcpp;

// Fitness function and prior
arma::vec fitness(const arma::uvec& N_k, const arma::vec& T_k)
{
  // eq. 19 from Scargle 2012
  arma::vec N_k_d = arma::conv_to<arma::vec>::from(N_k);
  return N_k_d % (arma::log(N_k_d) - arma::log(T_k));
}

double prior(const int N, const int Ntot)
{
  double p0(0.05); // Default
  // eq. 21 from Scargle 2012
  return 4.0 - std::log(73.53 * p0 * std::pow((double)N, -0.478));
}

// Compute the Bayesian Block bin edges for a single
// expression profile
arma::vec get_bb_bin_edges(const arma::vec& expr_prof)
{
  // Change R objects to armadillo objects
  int n_samples(expr_prof.n_rows);

  //Rcout << "t = " << t.t() << std::endl;
  
  arma::uvec unq_inds = arma::find_unique(expr_prof, false);
  arma::vec unq_vals = expr_prof.elem(unq_inds);
  int n_unq_vals(unq_inds.n_rows);
  //Rcout << "Unique values are " << unq_vals.t() << std::endl;
  //Rcout << "Unique indices are " << unq_inds.t() << std::endl;
  
  arma::uvec unq_inv = arma::uvec(n_samples);
  
  for(int i(0); i<n_samples; ++i)
  {
    unq_inv(i) = arma::as_scalar(arma::find(unq_vals==expr_prof[i]));
  }
  
  //Rcout << "Unique inverse is " << unq_inv.t() << std::endl;
  
  arma::uvec x = arma::uvec(n_unq_vals, arma::fill::zeros);
  
  if(n_samples==n_unq_vals)
    x = arma::uvec(arma::size(expr_prof), arma::fill::ones);
  else
  {
    for(int i(0); i<n_unq_vals; ++i)
    {
      arma::uvec tmp = arma::find(unq_inv==i);
      x(i) = arma::as_scalar(tmp.n_rows);
    }
    
  }
  arma::vec t = unq_vals;
  double sigma(1.0);
  
  //Rcout << "t = " << t.t() << std::endl;
  //Rcout << "x = " << x.t() << std::endl;
  
  int N(t.n_rows);
  
  arma::vec edges = arma::join_cols(t.row(0),
                                    0.5 * (t.rows(1,N-1) + t.rows(0, N-2)));
  edges = arma::join_cols(edges, t.row(N-1));
  
  //Rcout << "edges = " << edges.t() << std::endl;
  
  arma::vec block_length = t(N-1) - edges;
  
  //Rcout << "block length = " << block_length.t() << std::endl;
  
  // To store best configuration
  arma::vec best = arma::vec(N, arma::fill::zeros);
  arma::Col<int> last = arma::Col<int>(N, arma::fill::zeros);
  
  //Rcout << "best = " << best.t() << std::endl;
  
  // Start with first data cell; add one cell at each iteration
  
  for(int R(0); R<N; ++R)
  {
    //Rcout << "R=" << R << std::endl;
    arma::vec T_k = block_length.rows(0,R) - block_length[R+1];
    //Rcout << "T_k = " << T_k.t() << std::endl;
    
    arma::uvec N_k = arma::flipud(arma::cumsum(arma::flipud(x.rows(0,R))));
    
    //Rcout << "N_k = " << N_k.t() << std::endl;
    
    // Evaluate fitness function
    arma::vec fit_vec = fitness(N_k, T_k);
    //Rcout << "fit_vec = " << fit_vec.t() << std::endl;
    
    arma::vec A_R = fit_vec - prior(R+1, N);
    //Rcout << "A_R.rows(1,A_R.n_rows-1) = " << A_R.rows(1,A_R.n_rows-1) << std::endl;
    //Rcout << "best.rows(0,R-1) = " << best.rows(0,R-1) << std::endl;
    if(R==0)
    {
      A_R[0] += best[0];
    }
    else
      A_R.rows(1,A_R.n_rows-1) += best.rows(0,R-1);
    
    //Rcout << "A_R = " << A_R.t() << std::endl;
    
    int i_max = arma::index_max(A_R);
    last(R) = i_max;
    best(R) = A_R(i_max);
    
    //Rcout << "i_max = " << i_max << std::endl;
    
    //Rcout << "\n\n" << std::endl;
  }
  
  //Rcout << "best = " << best.t() << std::endl;
  //Rcout << "last = " << last.t() << std::endl;
  
  // Now find changepoints by iteratively peeling off the last block
  arma::Col<int> change_points = arma::Col<int>(N, arma::fill::zeros);
  int i_cp(N), ind(N);
  
  while(true)
  {
    --i_cp;
    change_points(i_cp) = ind;
    if(ind==0)
      break;
    ind = last(ind-1);
  }
  arma::Col<int> change_points_new = change_points.rows(i_cp, change_points.n_rows-1);
  arma::uvec cp_idxs = arma::conv_to<arma::uvec>::from(change_points_new);
  
  //Rcout << "i_cp = " << i_cp << "\tind = " << ind << std::endl;
  //Rcout << "change_points_new = " << change_points_new.t() << std::endl;
  
  return edges.elem(cp_idxs);
}

// Discretise a dataset using Bayesian Blocks algorithm
// [[Rcpp::export]]
arma::Mat<int> disc_dataset_bb_cpp(NumericMatrix expr_data, const int n_cores)
{
  const arma::mat data = R2armaMat_num(expr_data);
  const int n_genes(data.n_cols), n_samples(data.n_rows);
  
  // Compute bin edges for each gene in parallel
  arma::Mat<int> disc_data = arma::Mat<int>(arma::size(data),
                                            arma::fill::zeros);
  
  // Number of cores to use
  omp_set_num_threads(n_cores);
  
  int i;
  arma::vec bin_edges;
  
  #pragma omp parallel for shared(disc_data) private(i,bin_edges) schedule(auto) default(none)
  for(i=0; i<n_genes; ++i)
  {
    bin_edges = get_bb_bin_edges(data.col(i));
    for(int k(0); k<n_samples; ++k)
    {
      disc_data(k,i) = arma::find(data(k,i)>= bin_edges).max();
    }
  }
  
  // Change from C++ to R indexing
  disc_data += 1;
  
  return disc_data;
}

// Returns the Bayesian Block bin edges for a single gene
// [[Rcpp::export]]
arma::vec get_bb_bin_edges_cpp(NumericVector expr_prof)
{
  const arma::vec data = R2armaVec_num(expr_prof);
  const int n_samples(data.n_rows);
  return get_bb_bin_edges(data);
}



