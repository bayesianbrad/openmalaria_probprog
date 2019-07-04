# %%
import pyprob
from pyprob import Model
from pyprob.distributions import Poisson, Categorical, Normal, TruncatedNormal, Beta, Uniform, LogNormal
import torch as th
import torch
import math

import matplotlib.pyplot as plt
# % matplotlib
# inline

# http://www.ajtmh.org/docserver/fulltext/14761645/75/2_suppl/0750019.pdf?expires=1559041639&id=id&accname=guest&checksum=63084BE5CD7723A6025084787A900053


class ToyMalaria(Model):
    def __init__(self, pop_size=1050, likelihood_stddev=0.1, EIR=None):
        super().__init__(name='Toy Malaria')
        self.pop_size = pop_size
        self.likelihood_stddev = likelihood_stddev
        self.t_max = 365 * 2 # not totally clear
        self.days_per_step = 1
        #self.EIR = th.FloatTensor(self.t_max * self.days_per_step).zero_() + 67.0/365.0
        self.EIR = th.FloatTensor(self.t_max * self.days_per_step).zero_() + 5.5 / 365.0 # Matsari, intervention phase
        self.EIR_background = th.FloatTensor([67.0/365.0])
        self.A_max = 1.63
        self.prevalence_bins_upper = torch.FloatTensor([0.246746962983009,
                                             0.727755459545533,
                                             1.44479521753634,
                                             2.88435310594495,
                                             4.81735572357366,
                                             7.82460584536638,
                                             12.087165127225,
                                             16.8889599400757,
                                             24.2654245103456,
                                             33.7166980261767,
                                             48.1735572357366,
                                             67.3111742713562]) * 365
        self._eval = False
        self.n_age_bands = 7
        self.only_warmup = False

    def forward(self, _eval=False, use_pyprob=True):
        '''
        Parameters:
            i - Individual
            t - Individual at time t
            a - age group of individual at time t
            A(a(i,t)) - Mean body suface area

        Return:
            E_a(i,t) - The estimate at each time, for an individual i of
            number of infected bites recieved, adjusted for age.

        Latent variables:



        '''

        age_bands_lower = [0,1,5,9,19,29,44][:self.n_age_bands]
        age_bands_upper = [1,4,8,18,28,43,67][:self.n_age_bands]

        if use_pyprob:
            # Generate initial population age distribution
            pop_age_days = pyprob.sample(Uniform(low=torch.FloatTensor(age_bands_lower*int(self.pop_size//self.n_age_bands))*365.0,
                                                 high=torch.FloatTensor(age_bands_upper*int(self.pop_size//self.n_age_bands))*365.0)).floor()
        else:
            pop_age_days = Uniform(low=torch.FloatTensor(age_bands_lower*int(self.pop_size//self.n_age_bands))*365.0,
                                   high=torch.FloatTensor(age_bands_upper*int(self.pop_size//self.n_age_bands))*365.0).sample()

        if self._eval == False:
            # From force of infection model
            S_inf = pyprob.sample(TruncatedNormal(mean_non_truncated=0.01, stddev_non_truncated=0.05, low=0, high=1,
                                                  clamp_mean_between_low_high=True), name='S_inf')
            E_crit = pyprob.sample(TruncatedNormal(mean_non_truncated=0.040, stddev_non_truncated=0.035, low=0, high=1000,
                                                   clamp_mean_between_low_high=True), name='E_crit')
            S_imm = pyprob.sample(TruncatedNormal(mean_non_truncated=0.14, stddev_non_truncated=0.02, low=0, high=1,
                                                  clamp_mean_between_low_high=True), name='S_imm')
            X_p_crit = pyprob.sample(Normal(loc=1000, scale=10), name='Xpcrit')
            gamma_p = pyprob.sample(TruncatedNormal(mean_non_truncated=3.5, stddev_non_truncated=3, low=0, high=1000,
                                                    clamp_mean_between_low_high=True), name='gamma_p')

            # New parameters
            Y_h_star = pyprob.sample(TruncatedNormal(mean_non_truncated=10000000000, #should be infty
                                                     stddev_non_truncated=0.01,
                                                     low=10000000000,
                                                     high=10000000001,
                                                     clamp_mean_between_low_high=True),
                                     name='Y_h_star')
            X_y_star = pyprob.sample(TruncatedNormal(mean_non_truncated=3.5E7,
                                                     stddev_non_truncated=0.7E7,
                                                     low=0,
                                                     high=1000,
                                                     clamp_mean_between_low_high=True),
                                     name='X_y_star')
            X_h_star = pyprob.sample(TruncatedNormal(mean_non_truncated=97.3,
                                                     stddev_non_truncated=378.2,
                                                     low=0,
                                                     high=1000,
                                                     clamp_mean_between_low_high=True),
                                     name='X_h_star')
            alpha_m = pyprob.sample(TruncatedNormal(mean_non_truncated=0.9,
                                                    stddev_non_truncated=0.03,
                                                    low=0,
                                                    high=1.5,
                                                    clamp_mean_between_low_high=True),
                                    name='alpha_m')
            alpha_m_star = pyprob.sample(TruncatedNormal(mean_non_truncated=2.53,
                                                         stddev_non_truncated=0.22,
                                                         low=0,
                                                         high=5,
                                                         clamp_mean_between_low_high=True),
                                         name='alpha_m_star')
            D_x = pyprob.sample(TruncatedNormal(mean_non_truncated=0,
                                                stddev_non_truncated=0.1,
                                                low=0,
                                                high=1,
                                                clamp_mean_between_low_high=True),
                                name='D_x')
            sigma2_i = pyprob.sample(TruncatedNormal(mean_non_truncated=10.2,
                                                           stddev_non_truncated=0.5,
                                                           low=0,
                                                           high=15,
                                                           clamp_mean_between_low_high=True),
                                           name='sigma2_i')
            sigma2_0 = pyprob.sample(TruncatedNormal(mean_non_truncated=0.66,
                                                       stddev_non_truncated=0.13,
                                                       low=0,
                                                       high=1,
                                                       clamp_mean_between_low_high=True),
                                           name='sigma2_0')
            X_nu_star = pyprob.sample(TruncatedNormal(mean_non_truncated=0.92,
                                                     stddev_non_truncated=0.11,
                                                     low=0,
                                                     high=2,
                                                     clamp_mean_between_low_high=True),
                                     name='X_nu_star')
            nu_0 = pyprob.sample(TruncatedNormal(mean_non_truncated=4.8,
                                                 stddev_non_truncated=0.58,
                                                 low=0,
                                                 high=10,
                                                 clamp_mean_between_low_high=True),
                                 name='nu_0')
            nu_1 = pyprob.sample(TruncatedNormal(mean_non_truncated=0.18,
                                                 stddev_non_truncated=0.01,
                                                 low=0,
                                                 high=1,
                                                 clamp_mean_between_low_high=True),
                                 name='nu_1')

        else:
            # From force of infection model
            S_inf = self._eval["S_inf"]
            E_crit = self._eval["E_crit"]
            S_imm = self._eval["S_imm"]
            X_p_crit = self._eval["X_p_crit"]
            gamma_p = self._eval["gamma_p"]

            # new parameters
            Y_h_star = self._eval["Y_h_star"]
            X_y_star = self._eval["X_y_star"]
            X_h_star = self._eval["X_h_star"]
            alpha_m = self._eval["alpha_m"]
            alpha_m_star = self._eval["alpha_m_star"]
            D_x = self._eval["D_x"]
            sigma2_i = self._eval["sigma2_i"]
            sigma2_0 = self._eval["sigma2_0"]
            X_nu_star = self._eval["X_nu_star"]
            nu_0 = self._eval["nu_0"]
            nu_1 = self._eval["nu_1"]

        # From force of infection model
        # Initialize X_p
        E_max = self.EIR
        X_p = (pop_age_days * self.EIR_background).float() # WARM UP for force of infection
        pop_infected_new = torch.LongTensor(self.pop_size).zero_() # h in Eq. 5
        pop_infected_cum = torch.LongTensor(self.pop_size).zero_() # X_h, as given by Eq. 5

        # new parameters
        # assign d(i) to each member of the population (parasite density multiplier)
        l_p = 3 # number of 5-day steps for hepatic stage of infections
        # TODO: BATCH SAMPLING!
        if use_pyprob:
            pop_d_multiplier = pyprob.sample(LogNormal(loc=[1.0]*self.pop_size,
                                                       scale=[sigma2_i]*self.pop_size)) # equiv. to d_i (Eq 2)
        else:
            pop_d_multiplier = LogNormal(loc=[1.0]*self.pop_size,
                                         scale=[sigma2_i]*self.pop_size).sample() # equiv. to d_i (Eq 2)

        y_density = [[] for _ in range(self.pop_size)] # no infections initially
        y_tau_max = [[] for _ in range(self.pop_size)] # no infections initially
        y_tau = [[] for _ in range(self.pop_size)] # no infections initially, starts at -l_p
        X_y = [[] for _ in range(self.pop_size)]  # no infections initially, Eq. 3

        # total parasite densities at time t
        Y = th.FloatTensor([sum([ d_i_j for d_i_j in y_density[d_i]]) for d_i in range(self.pop_size)])
        Y_cum = th.FloatTensor([Y[d_i] for d_i in range(self.pop_size)])

        def _get_tau_max(i, use_pyprob=True):  # Eq. 1
            if use_pyprob:
                tau_max = math.exp(pyprob.sample(Normal(loc=5.13,
                                                        scale=0.8)))
            else:
                tau_max = math.exp(Normal(loc=5.13, scale=0.8).sample())

            return tau_max

        def _ln_y_G(y_tau, y_tau_max):
            # from Eq. 2 - this is an estimate, paper does not define this function (except for Figure 2)
            # very crude fit on Figure 1 - lols.
            a = max(0.05 * y_tau_max + 5.1, 4.4)
            c = a / (1 + y_tau_max / 18.0)
            b = (math.log(a / c)) / y_tau_max
            res = a * math.exp(-b * y_tau) - nu_0 #NOTE: use nu_1 for non-Garki scenarios!
            return res

        def _get_density(i, j, _y_tau, _y_tau_max, _y_density, _pop_d_multiplier, _X_y, _X_h, _pop_age_days, use_pyprob=True):

            if _y_tau[i][j] <= _y_tau_max[i][j] and _y_tau[i][j] >= 0:
                E_ln_y_0_i_j_tau = th.log(_pop_d_multiplier[i]) + \
                                   _ln_y_G(_y_tau[i][j], _y_tau_max[i][j])  # Eq. 2
                D_y = 1 / (1 + _X_y[i][j] / X_y_star)
                # X_h = pop_infected_cum  # Eq. 5
                D_h = 1 / (1 + _X_h[i] / X_h_star)  # Eq. 6
                D_m = 1 - alpha_m * th.exp(-0.693*(_pop_age_days[i]/365.0) / alpha_m_star)  # Eq. 7 - I assume "a" stands for "unit year"
                # Determine multiplicity of concurrent infections M_t
                M_t = sum([1 for j in range(len(_y_density[i])) if _y_tau[i][j] <= _y_tau_max[i][j] and _y_tau[i][j] >= 0])
                if M_t == 1:
                    E_ln_y_i_j_tau = D_y * D_h * D_m * E_ln_y_0_i_j_tau  # Eq. 9
                elif M_t > 1:
                    E_ln_y_i_j_tau = D_y * D_h * D_m * E_ln_y_0_i_j_tau + th.log(D_x / M_t + 1 - D_x)  # Eq. 9
                sigma2_y_i_j_tau = sigma2_0 / (1 + _X_h[i] / X_nu_star)  # Eq. 10
                if use_pyprob:
                    y_i_j = math.exp(pyprob.sample(Normal(loc=E_ln_y_i_j_tau,
                                                          scale=sigma2_y_i_j_tau)))
                else:
                    y_i_j = math.exp(Normal(loc=E_ln_y_i_j_tau,
                                            scale=sigma2_y_i_j_tau).sample())

                return y_i_j
            else:
                return y_density[i][j]

        def main_sim(t_max,
                     pop_age_days,
                     Y,
                     y_tau,
                     y_density,
                     pop_infected_cum,
                     Y_cum,
                     X_y,
                     const_EIR=None):

            all_born = False # whether all the population is already born
            for t in range(1, t_max):

                # only population that has been born yet
                if not all_born:
                    active_pop_set = (pop_age_days>=0).nonzero()
                    active_pop_set_th = (pop_age_days >= 0) #th.LongTensor(active_pop_set)
                    if active_pop_set.numel() == self.pop_size:
                        all_born = True
                else:
                    if self.only_warmup:
                        break
                    active_pop_set = list(range(self.pop_size))
                    active_pop_set_th = slice(None)

                # update age-adjusted EIR
                E_a = (E_max[t] if const_EIR is None else const_EIR) / self.A_max * self.get_mean_body_surface_area(pop_age_days[active_pop_set_th])

                X_p[active_pop_set_th] += E_a  # update X_p

                # construct S_1
                S_1 = S_inf + (1 - S_inf) / (1 + E_a / E_crit)

                # construct S_2

                S_2 = S_imm + (1 - S_imm) / (1 + (X_p[active_pop_set_th] / X_p_crit) ** gamma_p)

                # Probability of survival function
                S_p = S_1 * S_2

                # construct lambda
                _lambda = S_p * E_a

                # increase population age
                pop_age_days += 1

                # Number of infections generated in unit time per individual sampled from poisson
                if use_pyprob:
                    h = pyprob.sample(Poisson(_lambda), name='nInfections')
                else:
                    h = Poisson(_lambda).sample()

                # Liver-stage infections (IGNORED FOR NOW)
                # S_h = 1 / (1 + Y[active_pop_set_th] / Y_h_star)

                # approximation as multiple infections handled badly (but multiple infections are very unlikely in 1-day window)
                #try:
                #    h[h == 1][th.randperm(h[h == 1].shape[0])[:int(h[h == 1].shape[0] * S_h)]] = 0
                #except:
                #    pass

                pop_infected_new[active_pop_set_th] += h.long()

                if t % 5 == 1:
                    # update infection taus
                    y_tau = [[y_tau_i_j + 1 for y_tau_i_j in y_tau_i] for y_tau_i in y_tau]

                    # remove finished infections
                    def rm_inf(i, lst_idx):
                        y_tau[i] = [x for j, x in enumerate(y_tau[i]) if j not in lst_idx]
                        y_density[i] = [x for j, x in enumerate(y_density[i]) if j not in lst_idx]
                        y_tau_max[i] = [x for j, x in enumerate(y_tau_max[i]) if j not in lst_idx]

                    [rm_inf(i, [ j for j in range(len(y_density[i])) if y_tau[i][j] > y_tau_max[i][j] ]) for i in active_pop_set]

                    # DEBUG
                    # if pop_infected_new[active_pop_set_th].nonzero().numel() > 0:
                    #     k = 5
                    #     pass

                    # generate new infections
                    [y_density[active_pop_set[idx]].append(-1) for idx in pop_infected_new[active_pop_set_th].nonzero()]
                    [y_tau[active_pop_set[idx]].append(-l_p) for idx in pop_infected_new[active_pop_set_th].nonzero()]
                    [y_tau_max[active_pop_set[idx]].append(_get_tau_max(active_pop_set[idx], use_pyprob=use_pyprob)) for idx in pop_infected_new[active_pop_set_th].nonzero()]
                    [X_y[active_pop_set[idx]].append(Y[active_pop_set[idx]]) for idx in pop_infected_new[active_pop_set_th].nonzero()]
                    pop_infected_new[active_pop_set_th] = torch.LongTensor(active_pop_set.numel() if not all_born else self.pop_size).zero_()  # reset

                    # update parasite densities
                    # i, j, _y_tau, _y_tau_max, _y_density, _pop_d_multiplier, _X_y, _X_h, _pop_age_days)
                    y_density = [([ _get_density(i = a,
                                                 j = j,
                                                 _y_tau = y_tau,
                                                 _y_tau_max = y_tau_max,
                                                 _y_density = y_density,
                                                 _pop_d_multiplier = pop_d_multiplier,
                                                 _X_y = X_y,
                                                 _X_h = pop_infected_cum,
                                                 _pop_age_days = pop_age_days,
                                                 use_pyprob=use_pyprob) for j, _ in enumerate(y_density[a]) if (y_tau[a][j] <= y_tau_max[a][j] and y_tau[a][j] >= 0)] if pop_age_days[a] >= 0  else []) for a in range(self.pop_size)]

                    # Calculate total parasite densities (Eq 12)
                    Y[active_pop_set_th] = th.FloatTensor([sum([d_i_j for j, d_i_j in enumerate(y_density[i]) if y_tau[i][j] <= y_tau_max[i][j] and y_tau[i][j] >= 0 ]) \
                         for i in active_pop_set])

                    # calculate Y_cum (needed for Eq. 3)
                    Y_cum[active_pop_set_th] += Y[active_pop_set_th]

                    # Calculate cumulative density of asexual parasitemia (Eq. 3)
                    X_y = [[(X_y[i][j] + Y[i] - y_density[i][j] if y_tau[i][j] <= y_tau_max[i][j] else 0) for j in
                            range(len(y_density[i]))] for i in range(self.pop_size)]

                pop_infected_cum[active_pop_set_th] += pop_infected_new[active_pop_set_th].long()  # Eq. 5

        # DEBUG
        # for i in range(self.pop_size):
        #    for j in range(len(y_density[i])):
        #        b = (X_y[i][j] + Y[i][j] - y_density[i][j] if y_tau[i][j] <= y_tau_max[i][j] else -1

        #    [[(X_y[i][j] + Y[i][j] - y_density[i][j] if y_tau[i][j] <= y_tau_max[i][j] else -1) for j in
        #      range(len(y_density[i]))] for i in range(self.pop_size)]

        # WARM UP
        max_age = pop_age_days.max().item()
        t_max = max_age
        pop_age_days -= max_age
        main_sim(int(t_max),
                 pop_age_days,
                 Y,
                 y_tau,
                 y_density,
                 pop_infected_cum,
                 Y_cum,
                 X_y,
                 const_EIR=self.EIR_background)

        t_max = self.t_max
        main_sim(int(t_max),
                 pop_age_days,
                 Y,
                 y_tau,
                 y_density,
                 Y_cum,
                 X_y,
                 pop_infected_cum)

        # TODO: prevalence now derives from parasite densities, taking detection thresholds into account!
        prevalence = th.FloatTensor(self.prevalence_bins_upper.shape[0]).zero_()
        age_band = th.FloatTensor(self.prevalence_bins_upper.shape[0]).zero_()
        detection_threshold = 2 # 40 for non-Garki sites (see Fig. Table 2)
        for i in range(self.prevalence_bins_upper.shape[0]):
            prev_band = (pop_age_days > (0 if i == 0 else self.prevalence_bins_upper[i - 1])) & (
                    pop_age_days < self.prevalence_bins_upper[i])
            #if pop_infected[prev_band].sum() != 0:  # _lambda_cum:
            #    prevalence[i] = pop_infected[prev_band & (pop_infected == 1)].float().sum() / prev_band.sum()
            if prev_band.sum() != 0:
                prevalence[i] = Y[prev_band & Y > detection_threshold].float().sum() / prev_band.sum()
            age_band[i] = prev_band.sum()

        if use_pyprob:
            pyprob.tag(value=prevalence, name='prevalance')

        return prevalence  # The thing we want to do inference over


    def get_mean_body_surface_area(self, pop_age_days):
        """
        We use a crude estimate based on a quintic regression
        based on a fit on data points:
        1. 3kg, 50cm=0.20m^2 at birth (age=0)
        2. 24kg, 127cm=0.92m^2 at (age=10years)
        3. 60kg, 160cm=1.63m^2 at (age=20years, limit)
        These data points are roughly based on Nigerian population,
        "Relative Height and Weight among Children and Adolescents of Rural Southwestern Nigeria", Ayoola et al, 2009
        """
        # adjust EIR according to population age
        pop_age_days_capped = pop_age_days.clone().float()
        pop_age_days_capped[pop_age_days_capped > 365 * 20] = 365 * 20
        a = 0.2
        b = -5.670374E-9
        c = 5.670316E-8
        d = 6.500411E-12
        e = -2.509808E-15
        f = 1.450615E-19
        A = a + \
            b * pop_age_days_capped + \
            c * (pop_age_days_capped ** 2) + \
            d * (pop_age_days_capped ** 3) + \
            e * (pop_age_days_capped ** 4) + \
            f * (pop_age_days_capped ** 5)
        return A


model = ToyMalaria()
model.warmup_only = True
model.n_age_bands = 3
from functools import partial
model._eval = {"S_inf":th.FloatTensor([0.049]),
               "E_crit":th.FloatTensor([0.032]),
               "S_imm":th.FloatTensor([0.12]),
               "X_p_crit":th.FloatTensor([523.0]),
               "gamma_p":th.FloatTensor([5.1]),
               "Y_h_star":th.FloatTensor([10000000.0]), # infinity
               "X_y_star":th.FloatTensor([3.5E7]),
               "X_h_star":th.FloatTensor([97.3]),
               "alpha_m":th.FloatTensor([0.9]),
               "alpha_m_star":th.FloatTensor([2.53]),
               "D_x": th.FloatTensor([0]),
               "sigma2_i": th.FloatTensor([10.2]),
               "sigma2_0": th.FloatTensor([0.66]),
               "X_nu_star": th.FloatTensor([0.92]),
               "nu_0": th.FloatTensor([4.8]),
               "nu_1": th.FloatTensor([0.18]),
            }

# compare prevalence to field data and immunity model
import numpy as np
prevalence = model.forward(use_pyprob=False)
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
plt.plot(prevalence.cpu().numpy())
plt.plot(obs)
plt.plot(toy1)
plt.show()
plt.savefig("out_parasite.png")

prior_traces = model.prior_traces(num_traces=1)

quit()
prior_traces = model.prior_traces(num_traces=10)
prior_population = prior_traces.map(lambda trace: trace.named_variables['prevalance'].value)


ground_truth_trace = next(model._trace_generator())
ground_truth_prevalence = ground_truth_trace.named_variables['prevalance'].value
print(ground_truth_prevalence)
plt.plot(ground_truth_prevalence.numpy())

is_posterior_traces = model.posterior_traces(observe={'prevalance': ground_truth_prevalence}, num_traces=10)
is_posterior_sinf = is_posterior_traces.map(lambda trace: trace.named_variables['S_inf'].value)
plt.plot(ground_truth_prevalence.numpy())
plt.plot(is_posterior_sinf.mean.numpy())

