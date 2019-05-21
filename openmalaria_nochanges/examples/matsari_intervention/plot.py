from matplotlib import pyplot as plt
import numpy as np

age_pop = None
age_inf = None
with open("output.txt", "r") as f:
    tmp = list(f.readlines())
    vals = list(map(lambda x:float(x.split("	")[-1]), tmp))
    age_pop = vals[:13]
    age_inf = vals[13:]

om = np.array(age_inf) / np.array(age_pop)
toy1 = np.array([0.1228, 0.1234, 0.1272, 0.1428, 0.1760, 0.2267, 0.1642, 0.1047, 0.0928,
        0.0892, 0.0882, 0.0881])
obs = np.array([0.001156069364162,
       0.018208092485549,
       0.050289017341041,
       0.135260115606936,
       0.038150289017341,
       0.165606936416185,
       0.172254335260116,
       0.053757225433526,
       0.038439306358382,
       0.032947976878613,
       0.039595375722543,
       0.047109826589595])
plt.plot(om)
plt.plot(obs)
plt.plot(toy1)
plt.show()
plt.savefig("out.png")
