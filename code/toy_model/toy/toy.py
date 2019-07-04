import os

from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import torch as th


class ToyMalaria(Model):
    def __init__(self, pop_size=10, likelihood_stddev=0.1, EIR=None):
        super().__init__(name='Toy Malaria')
        self.pop_size = pop_size
        self.likelihood_stddev = likelihood_stddev
        self.t_max = 10
        self.days_per_step = 1
        self.EIR = EIR
        self.A_max = 1.63 # average surface area of population >= 20 years
        # Note: This is currently crudely estimated as 160cm, 60kg
        # by Mosteller: 1.63 m^2

    def forward(self):
        # Generate initial population age distribution
        pop_age_days = th.LongTensor(self.pop_size).random_(0, 80*365)
        E_a = None # equiv. to E_a
        E_max = self.EIR

        # Initialize X_p
        X_p = th.FloatTensor(self.pop_size).zero_() # equiv. to X_p, Equ. 6
        for t in range(max(pop_age_days)):
            X_p[pop_age_days >= t] += self.get_EIR_age_adjusted(pop_age_days[pop_age_days >= t]*0+t)

        # now evolve over time
        for t in range(self.t_max):
            # update age-adjusted EIR
            E_a = ( E_max[t] / self.A_max ) * self.get_mean_body_surface_area(pop_age_days) #self.get_EIR_age_adjusted(pop_age)
            X_p += E_a # update X_p

            # construct S_1
            S_1 = S_infinity + (1 - S_infinity) / (1 + E_a / E_star)

            # construct S_2
            S_2 = S_imm + (1 - S_imm) / (1 + (X_p / X_p_star) ** gamma_P)

            # construct S_p
            S_p = S_1 * S_2

            # construct lambda
            _lambda = S_p * E_a

            # increase population age
            pop_age_days += 1

        return

    def get_mean_body_surface_area(pop_age_days):
        """
        We use a crude estimate based on a quintic regression
        based on a fit on data points:
        1. 3kg, 50cm=0.20m^2 at birth (age=0)
        2. 24kg, 127cm=0.92m^2 at (age=10years)
        3. 60kg, 160cm=1.63m^2 at (age=20years, limit)
        4. 0.45m^2 at age=1
        These data points are roughly based on Nigerian population,
        "Relative Height and Weight among Children and Adolescents of Rural Southwestern Nigeria", Ayoola et al, 2009
        """
        # adjust EIR according to population age
        pop_age_days_capped = pop_age_days.clone()
        pop_age_days_capped[pop_age_days_capped > 365*20] = 365*20
        a = 0.2
        b = 0.0008474793
        c = -4.916789E-7
        d = 1.323107E-10
        e = -1.484702E-14
        f = 5.868189E-19
        A = a + \
            b*pop_age_days_capped + \
            c*(pop_age_days_capped**2) + \
            d*(pop_age_days_capped**3) + \
            e*(pop_age_days_capped**4) + \
            f*(pop_age_days_capped**5)
        return A

# MOD = None
# class PredatorPreyEnv(MultiAgentEnv):
#     global GridEnvModule
#
#     def __init__(self, **kwargs):
#         # Unpack arguments from sacred
#         args = kwargs["env_args"]
#         if isinstance(args, dict):
#             args = convert(args)
#
#         from torch.utils.cpp_extension import load
#         MOD = load(name="gridenv_cpp", sources=[os.path.join(os.path.dirname(__file__), "gridenv.cpp")], verbose=True)
#
#         self.grid_shape_x, self.grid_shape_y = args.predator_prey_shape
#         self.env = MOD.PredatorPreyEnv(5, self.grid_shape_x, self.grid_shape_y)
#         try:
#             self.env.Test()
#         except Exception as e:
#             print(e)
#             pass
#         pass

if __name__ == "__main__":
    kwargs = dict(env_args=dict(predator_prey_shape=(5,5)))
    a = PredatorPreyEnv(**kwargs)
    pass
