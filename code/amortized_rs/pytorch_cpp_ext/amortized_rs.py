from torch.utils.cpp_extension import load

amortized_rs = load(name="amortized_rs",
            sources=["amortized_rs.cpp"])

for i in range(100):
    a = amortized_rs.f()
    print(a)
