#include <torch/torch.h>

#include <iostream>
#include <limits>

using namespace at;

//https://pytorch.org/cppdocs/api/namespace_at.html#namespace-at
at::Tensor f(void);
Scalar R1(Scalar z_1, unsigned int i);
Scalar R2(Scalar z_2, Scalar z_3);
Scalar R3(Scalar z_5, unsigned int M);
Scalar f(Scalar x, auto mu1, auto sigma1, auto mu2, auto sigma2)
torch::Tensor batch_f(unsigned int bs){
    auto ret = torch::empty({bs, 6}, kFloat);
    for(int i=0;i<bs;i++){
        ret[i] = f();
    }
    return ret;
}

at::Tensor f(void) {
  Scalar z_1 = at::empty({1}, kFloat).normal_(0.0, 1.0).item();
  Scalar z_2;

  int ctr = 0;
  do {
    z_2 = R1(z_1, 0);
    ctr += 1;
  } while (z_2.to<float>() == std::numeric_limits<float>::infinity());
  //std::cout << ctr;
  Scalar z_3 = at::empty({1}, kFloat).uniform_(0,2).item();
  Scalar z_4 = R2(z_2, z_3);
  Scalar z_5 = at::empty({1}, kFloat).normal_(30.0,50.0);
  Scalar z_6 = R3(z_5)
  //std::cout << z_1 << "|" << z_2 << "|" << z_3 << "|" << z_4;
  //float data[] = {z_1.to<float>(), z_2.to<float>(), z_3.to<float>(), z_4.to<float>()};
  auto ret = torch::empty({6}, kFloat);
  ret[0] = z_1;
  ret[1] = z_2;
  ret[2] = z_3;
  ret[3] = z_4;
  ret[4] = z_5;
  ret[5] = z_6;
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
        temp = at::empty({1}, kFloat).normal_(z_1.to<float>(), 1.0).item();
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

Scalar f(x, mu1, sigma1, mu2, sigma2) {

}
    auto const1 =  1 / (2 * np.pi * sigma1 ** 2 ) ** 0.5
    auto const2 =  1 / (2 * np.pi * sigma2 ** 2 ) ** 0.5
    auto body1 =  torch.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2 ))
    auto body2 =  torch.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2 ))
    return  const1*body1 + const2*body2

def g(x,mu1, sigma1):
    const = 1 / (2 * np.pi * sigma1 ** 2 ) ** 0.5
    body = torch.exp(-(x - mu1)** 2 / (2 * sigma1 ** 2 ))
    return const * body

Scalar R2(Scalar z_2, Scalar z_3) {
    Scalar temp = at::empty({1}, kFloat).normal_(z_3.to<float>(), 1.0).item();
    while (temp.to<float>()  < z_2.to<float>() ) {
        temp = at::empty({1}, kFloat).normal_(z_3.to<float>(), 1.0).item();
    }
    return temp;
}

Scalar R3(Scalar z_4){
       Scalar temp =
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("f", &f, "f");
  m.def("batch_f", &batch_f, "batch_f");
}