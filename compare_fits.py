from tsodyks_markram_plasticity_fit_NMDA import Rothman_NMDA_STP
from tsodyks_markram_plasticity_fit_NMDA import plot_optimisation_results as plot_NMDA
from tsodyks_markram_plasticity_fit_AMPA import Rothman_AMPA_STP
from tsodyks_markram_plasticity_fit_AMPA import plot_optimisation_results as plot_AMPA

AMPA = True
NMDA = True

if __name__ == '__main__':
    if AMPA:
        # -- AMPA --
        problem_AMPA = Rothman_AMPA_STP()
        candidate_AMPA = [0.5389, 0.2014, 1.736, 0.6395, 0.02100, 0.3426, 2.418, 17., 0.2703, 80.13, 0.2301, 0.2096, 0.1157, 0.05047, 0.5, 7.035, 45.0, 0.6742, 4.255]
        fitness_AMPA = 0.5656
        max_evaluations_AMPA = 21000
        pop_size_AMPA = 280

        plot_AMPA(problem_AMPA,
                  candidate_AMPA,
                  fitness_AMPA,
                  max_evaluations_AMPA,
                  pop_size_AMPA)
    if NMDA:
        # -- NMDA --
        problem_NMDA = Rothman_NMDA_STP()
        candidate_NMDA = [0.8666, 1.087, 2.308, 0.7102, 10.41, 75.03, 0.04270, 140.6, 10.00]
        fitness_NMDA = 1.148
        max_evaluations_NMDA = 4200
        pop_size_NMDA = 70

        plot_NMDA(problem_NMDA,
                  candidate_NMDA,
                  fitness_NMDA,
                  max_evaluations_NMDA,
                  pop_size_NMDA)


