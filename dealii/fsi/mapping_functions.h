namespace
{
template <int dim>
inline void GetF(std::vector<dealii::Tensor<2,dim>>& fs,
    const std::vector<std::vector<dealii::Tensor<1,dim>>>& sol_grads) {
  if (!fs.size()) fs.resize(sol_grads.size());
  for (int q=0; q<sol_grads.size(); ++q)
    for (int d1=0; d1<dim; ++d1)
      for (int d2=0; d2<dim; ++d2)
        fs[d1][d2] = (d1==d2) + sol_grads[q][dim+d1][d2];
}

template <int dim>
inline void GetFT(std::vector<dealii::Tensor<2,dim>>& f_ts,
    const std::vector<dealii::Tensor<2,dim>>& fs) {
  if (!f_ts.size()) f_ts.resize(fs.size());
  for (int q=0; q<fs.size(); ++q)
    f_ts[q] = dealii::transpose(fs[q]);
}

template <int dim>
inline void GetFTInv(std::vector<dealii::Tensor<2,dim>>& fs_inv,
    const std::vector<dealii::Tensor<2,dim>>& fs) {
  if (!fs_inv.size()) fs_inv.resize(fs.size());
  for (int q=0; q<fs_inv.size(); ++q)
    fs_inv[q] = dealii::invert(fs[q]);
}
}
