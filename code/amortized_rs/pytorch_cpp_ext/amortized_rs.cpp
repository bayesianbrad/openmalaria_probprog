#include <torch/torch.h>

#include <iostream>
#include <limits>

using namespace at;

//https://pytorch.org/cppdocs/api/namespace_at.html#namespace-at
Scalar R1(Scalar z_1, unsigned int i);
Scalar R2(Scalar z_2, Scalar z_3);


at::Tensor f(void) {
  Scalar z_1 = at::empty({1}, kDouble).normal_(0.0, 1.0).item();
  Scalar z_2;
  do {
    z_2 = R1(z_1, 0);
  } while (z_2.to<double>() == std::numeric_limits<double>::infinity());
  Scalar z_3 = at::empty({1}, kDouble).uniform_().item();
  Scalar z_4 = R2(z_2, z_3);
  //std::cout << z_1 << "|" << z_2 << "|" << z_3 << "|" << z_4;
  //double data[] = {z_1.to<double>(), z_2.to<double>(), z_3.to<double>(), z_4.to<double>()};
  auto ret = torch::empty({4}, kDouble);
  ret[0] = z_1;
  ret[1] = z_2;
  ret[2] = z_3;
  ret[3] = z_4;
  return ret; //torch::from_blob(data, {4});
}

Scalar R1(Scalar z_1, unsigned int i) {
    Scalar temp = at::empty({1}, kDouble).normal_(z_1.to<double>(), 1.0).item();
    if (temp.to<double>() > 0){
        return temp;
    } else if (i == 1000){
        return CPU(kDouble).scalarTensor(std::numeric_limits<double>::infinity()).item();
    } else {
        i += 1;
        return R1(z_1, i);
    }
}

Scalar R2(Scalar z_2, Scalar z_3) {
    Scalar temp = at::empty({1}, kDouble).normal_(z_3.to<double>(), 1.0).item();
    while (temp.to<double>()  < z_2.to<double>() ) {
        temp = torch::randint(1, {1}).normal_(z_3.to<double>(), 1.0).item();
    }
    return temp;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("f", &f, "f");
}