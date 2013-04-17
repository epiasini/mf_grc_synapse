from tsodyks_markram_plasticity_fit_NMDA import Rothman_NMDA_STP
from tsodyks_markram_plasticity_fit_NMDA import plot_optimisation_results as plot_NMDA
from tsodyks_markram_plasticity_fit_AMPA import Rothman_AMPA_STP
from tsodyks_markram_plasticity_fit_AMPA import plot_optimisation_results as plot_AMPA

AMPA = False
NMDA = True

if __name__ == '__main__':
    if AMPA:
        # -- AMPA --
        problem_AMPA = Rothman_AMPA_STP()
        candidate_AMPA = [0.3274, 4.492, 0.3659, 0.3351, 1.651, 0.1249, 131.0, 0.5548, 0.3000, 0.3376, 0.153, 0.4, 4.899, 43.10, 0.2792, 14.85]
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
        candidate_NMDA = [0.8647, 3.683, 0.5730, 13.52, 121.9, 0.03220, 236.1, 6.394]
        fitness_NMDA = 0.035
        max_evaluations_NMDA = 22800
        pop_size_NMDA = 140

        plot_NMDA(problem_NMDA,
                  candidate_NMDA,
                  fitness_NMDA,
                  max_evaluations_NMDA,
                  pop_size_NMDA)


