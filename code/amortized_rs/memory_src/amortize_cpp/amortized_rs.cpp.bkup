#include <torch/torch.h>

#include <iostream>
#include <limits>

using namespace at;

//https://pytorch.org/cppdocs/api/namespace_at.html#namespace-at
at::Tensor f(void);
Scalar R1(Scalar z_1, unsigned int i);
Scalar R2(Scalar z_2, Scalar z_3);

torch::Tensor batch_f(unsigned int bs){
    auto ret = torch::empty({bs, 4}, kFloat);
    for(int i=0;i<bs;i++){
        ret[i] = f();
    }
    return ret;
}

at::Tensor f(void) {
  Scalar z_1 = at::empty({1}, kFloat).normal_(0.0, 0.1).item();
  Scalar z_2;

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

Scalar R1(Scalar z_1, unsigned int i) {
    Scalar temp;
//    if (temp.to<float>() > 0){
//        return temp;
//    } else if (i == 100){
//        return CPU(kFloat).scalarTensor(std::numeric_limits<float>::infinity()).item();
//    } else {
//        i += 1;
//        return R1(z_1, i);
//    }
    long j = 0;
    while (true){
        temp = at::empty({1}, kFloat).normal_(z_1.to<float>(), 5).item();
        if (temp.to<float>() > 0){
            return temp;
        } else if (j == 10000){
            // return at::CPU(kFloat).scalarType(std::numeric_limits<float>::infinity()).item();
            //return at::from_blob({std::numeric_limits<float>::infinity()}, {1}).item();
            return Scalar(std::numeric_limits<float>::infinity());
        } else {
            j += 1;
        }
    }
}

Scalar R2(Scalar z_2, Scalar z_3) {
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