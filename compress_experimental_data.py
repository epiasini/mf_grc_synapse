# organise experimental data in hdf5 files. NMDA data is averaged
# across cells. NOTE: the traces are averaged over BOTH time and
# voltage, and no alignment is performed.
import h5py
import glob
import numpy as np

DATA_DIR = "/home/ucbtepi/doc/jason_data/JasonsIAFmodel/Jason_Laurence_AMPA_NMDA_Trains"
NMDA_DIR = DATA_DIR + "/NMDA"
AMPA_DIR = DATA_DIR + "/AMPA"
AMPA_FITS_DIR = DATA_DIR + "/Fits_gAMPA"
NMDA_FITS_DIR = DATA_DIR + "/Fits_gNMDA"
TIME_DIR = DATA_DIR + "/StimTimes"
frequencies = [5, 10, 20, 30, 50, 80, 100, 150]
n_protocols = 4


for syn_type in []:#["AMPA", "NMDA"]:
    with h5py.File("{0}_experimental_data_prova.hdf5".format(syn_type)) as data_repo:
        for freq in frequencies:
            f_group = data_repo.create_group(str(freq))
            for prot in range(n_protocols):
                p_group = f_group.create_group(str(prot))
                # load pulse times and waveforms from Jason's  txt files
                pulse_times = np.loadtxt(TIME_DIR + "/gp{0}_{1}hz_times.txt".format(prot,
                                                                                    freq))
                waveform_filenames = glob.glob(DATA_DIR + "/" + syn_type + "/*{0}_{1}hz_G{2}*.txt".format(syn_type, freq, prot))
                waveforms = np.array([np.loadtxt(f) for f in waveform_filenames])
                # truncate them so that their time axes coincide
                #max_start_time = waveforms[:,0,0].max()
                #min_end_time = waveforms[:,-1,0].min()
                #print max_start_time, min_end_time
                #truncated_waveforms = np.array([wf[np.searchsorted(wf[:,0], max_start_time):np.searchsorted(wf[:,0], min_end_time)] for wf in waveforms])
                #print truncated_waves
                # write hdf5 datasets
                p_group.create_dataset('pulse_times', data=pulse_times)
                p_group.create_dataset('average_waveform', data=np.mean(waveforms, axis=0))

with h5py.File("AMPA_jason_fit_traces.hdf5") as data_repo:
    for freq in []:#frequencies:
        f_group = data_repo.create_group(str(freq))
        for prot in range(n_protocols):
            p_group = f_group.create_group(str(prot))
            pulse_times = np.loadtxt(TIME_DIR + "/gp{0}_{1}hz_times.txt".format(prot, freq))
            waveform = np.loadtxt(AMPA_FITS_DIR + "/Fit_AMPA_{0}hz_G{1}.txt".format(freq, prot))
            # write hdf5 datasets
            p_group.create_dataset('pulse_times', data=pulse_times)
            p_group.create_dataset('average_waveform', data=waveform)

with h5py.File("NMDA_jason_fit_traces.hdf5") as data_repo:
    for freq in frequencies:
        f_group = data_repo.create_group(str(freq))
        for prot in range(n_protocols):
            p_group = f_group.create_group(str(prot))
            pulse_times = np.loadtxt(TIME_DIR + "/gp{0}_{1}hz_times.txt".format(prot, freq))
            waveform = np.loadtxt(NMDA_FITS_DIR + "/S_NMDA_{0}hz_G{1}_270104.txt".format(freq, prot))
            # write hdf5 datasets
            p_group.create_dataset('pulse_times', data=pulse_times)
            p_group.create_dataset('average_waveform', data=waveform)
