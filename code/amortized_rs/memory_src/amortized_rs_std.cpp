#include <torch/torch.h>

#include <iostream>
#include <limits>
#include <random>

using namespace at;

//https://pytorch.org/cppdocs/api/namespace_at.html#namespace-at
Tensor f(void);
Tensor R1(double z_1);
Tensor R2(double z_2, double z_3);


Tensor batch_f(unsigned int bs){
    auto ret = torch::empty({batch_size, 4}, kFloat);
    for(int i=0;i<bs;i++){
        ret_view = ret.narrow(i, 0, -1)
        ret_view = f()
    }
    return ret
}

inline double f(void) {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,1.0);
  double z_1 = distribution(generator);
  double z_2;
  int ctr = 0;
  do {
    z_2 = R1(z_1, 0);
    ctr += 1;
  } while (z_2.to<float>() == std::numeric_limits<float>::infinity());
  //std::cout << ctr;
  Scalar z_3 = at::empty({1}, kFloat).uniform_(0,2).item();
  Scalar z_4 = R2(z_2, z_3);
  //std::cout << z_1 << "|" << z_2 << "|" << z_3 << "|" << z_4;
  //float data[] = {z_1.to<float>(), z_2.to<float>(), z_3.to<float>(), z_4.to<float>()};
  auto ret = torch::empty({4}, kFloat);
  ret[0] = z_1;
  ret[1] = z_2;
  ret[2] = z_3;
  ret[3] = z_4;
  return ret; //torch::from_blob(data, {4});
}

inline double R1(double z_1) {
    Scalar temp;
    long j = 0;
    while (true){
        temp = at::empty({1}, kFloat).normal_(z_1.to<float>(), 1.0).item();
        if (temp.to<float>() > 0){
            return temp;
        } else if (j == 10000){
            return CPU(kFloat).scalarTensor(std::numeric_limits<float>::infinity()).item();
        } else {
            j += 1;
        }
    }
}

inline double R2(Scalar z_2, Scalar z_3) {
    Scalar temp = at::empty({1}, kFloat).normal_(z_3.to<float>(), 1.0).item();
    while (temp.to<float>()  < z_2.to<float>() ) {
        temp = at::empty({1}, kFloat).normal_(z_3.to<float>(), 1.0).item();
    }
    return temp;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("f", &f, "f");
  m.def("batch_f", &batch_f, "batch_f");
}