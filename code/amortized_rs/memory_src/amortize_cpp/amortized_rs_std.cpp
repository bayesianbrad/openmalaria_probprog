#include <torch/torch.h>

#include <iostream>
#include <limits>
#include <random>

using namespace at;

//https://pytorch.org/cppdocs/api/namespace_at.html#namespace-at
torch::Tensor f(void);
double R1(double z_1, std::default_random_engine &generator);
double R2(double z_2, double z_3, std::default_random_engine &generator);


torch::Tensor batch_f(unsigned int bs){
    auto ret = torch::empty({bs, 4}, kFloat);
    for(int i=0;i<bs;i++){
        auto ret_view = ret.narrow(0, i, 1);
        ret_view = f();
    }
    return ret;
}

inline torch::Tensor f(void) {
  std::random_device rd;
  std::default_random_engine generator( rd() );
  std::normal_distribution<double> distribution(0.0, 1.0);
  double z_1 = distribution(generator);
  double z_2;
  int ctr = 0;
  do {
    z_2 = R1(z_1, generator);
    ctr += 1;
  } while (z_2 == std::numeric_limits<float>::infinity());
  // double z_3 = at::empty({1}, kFloat).uniform_(0,2).item();
  std::uniform_real_distribution<double> uni_dist(0.0, 2.0);
  double z_3 = uni_dist(generator);
  double z_4 = R2(z_2, z_3, generator);
  auto ret = torch::empty({4}, kFloat);
  ret[0] = z_1;
  ret[1] = z_2;
  ret[2] = z_3;
  ret[3] = z_4;
  return ret;
}

inline double R1(double z_1, std::default_random_engine &generator) {
  std::normal_distribution<double> distribution(z_1, 1.0);
  long j = 0;
  while (true){
    //temp = at::empty({1}, kFloat).normal_(z_1.to<float>(), 1.0).item();
    auto temp = distribution(generator);
    if (temp > 0){
        return temp;
    } else if (j == 10000){
        return std::numeric_limits<float>::infinity();
    } else {
        j += 1;
    }
  }
}

inline double R2(double z_2, double z_3, std::default_random_engine &generator) {
    // Scalar temp = at::empty({1}, kFloat).normal_(z_3.to<float>(), 1.0).item();
    std::normal_distribution<double> distribution(z_3, 1.0);
    auto temp = distribution(generator);
    while (temp < z_2) {
        temp = distribution(generator);
    }
    return temp;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("f", &f, "f");
  m.def("batch_f", &batch_f, "batch_f");
}
