from tsodyks_markram_plasticity_fit_NMDA import Rothman_NMDA_STP
from tsodyks_markram_plasticity_fit_NMDA import plot_optimisation_results as plot_NMDA
from tsodyks_markram_plasticity_fit_AMPA import Rothman_AMPA_STP
from tsodyks_markram_plasticity_fit_AMPA import plot_optimisation_results as plot_AMPA

AMPA = True
NMDA = False

if __name__ == '__main__':
    if AMPA:
        # -- AMPA --
        problem_AMPA = Rothman_AMPA_STP()
        candidate_AMPA = [0.3275, 4.5, 0.5011, 0.0001329, 0.3493, 2.45, 23.0, 0.1260, 141.1, 0.2189, 0.1700, 0.3089, 0.123, 1.53, 7.0, 49.97, 0.2849, 12.66]
        fitness_AMPA = 0.021
        max_evaluations_AMPA = 21000
        pop_size_AMPA = 140

        plot_AMPA(problem_AMPA,
                  candidate_AMPA,
                  fitness_AMPA,
                  max_evaluations_AMPA,
                  pop_size_AMPA)
    if NMDA:
        # -- NMDA --
        problem_NMDA = Rothman_NMDA_STP()
        candidate_NMDA = [0.9501, 2.870, 0.8586, 10.31, 78.38, 0.03734, 178.34, 7.124]
        fitness_NMDA = 0.035
        max_evaluations_NMDA = 18000
        pop_size_NMDA = 90

        plot_NMDA(problem_NMDA,
                  candidate_NMDA,
                  fitness_NMDA,
                  max_evaluations_NMDA,
                  pop_size_NMDA)


