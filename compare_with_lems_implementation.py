from tsodyks_markram_plasticity_fit_AMPA import Rothman_AMPA_STP
from tsodyks_markram_plasticity_fit_AMPA import plot_lems_comparison as plot_AMPA

AMPA = True

if __name__ == '__main__':
    if AMPA:
        # -- AMPA --
        problem_AMPA = Rothman_AMPA_STP()
        candidate_AMPA = [0.3274, 4.492, 0.3659, 0.3351, 1.651, 0.1249, 131.0, 0.5548, 0.3000, 0.3376, 0.153, 0.4, 4.899, 43.10, 0.2792, 14.85]

        plot_AMPA(problem_AMPA,
                  candidate_AMPA)

