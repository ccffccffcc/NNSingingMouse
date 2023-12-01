import numpy as np
import os
import scipy.io as spio
from multiprocessing import Pool
from functools import partial
import scipy

alpha = 0.01  # significance level


def loadmat(filename):
    # https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _tolist(d[key])
        return d

    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


# Cache spike counts for faster processing
def cache_spike_counts(data_dir, **kwargs):
    cpu = kwargs.get("cpu", 1)
    pre_length = kwargs.get("pre_length", 0)
    post_length = kwargs.get("post_length", 0)
    alignment = kwargs.get("alignment", None)
    subtraction = kwargs.get("subtraction", False)

    audio_data_mat = spio.loadmat(os.path.join(data_dir, "BehavioralTimings.mat"))
    start_sing_time_array = audio_data_mat["T_Motor"][:, 0]
    end_sing_time_array = audio_data_mat["T_Motor"][:, 1]
    song_length_array = end_sing_time_array - start_sing_time_array
    sensory_array = audio_data_mat["T_Sensory"]

    neural_data_mat = spio.loadmat(os.path.join(data_dir, "clusterOutput.mat"))
    clusters_array = neural_data_mat["clusters"]

    if alignment is None:
        alignment = "on"
    if type(alignment) is np.ndarray:
        pass
    elif alignment == "on":
        alignment = start_sing_time_array
    elif alignment == "off":
        alignment = end_sing_time_array
    spike_time_list = []
    for ii in range(0, clusters_array[0].shape[0]):
        tempt = set()
        flag = 1
        for i in alignment:
            pre = i - pre_length
            post = i + post_length
            if flag:
                if subtraction:
                    tempt = set(
                        clusters_array[0][ii][0][
                            (pre < clusters_array[0][ii][0])
                            * (clusters_array[0][ii][0] < post)
                        ]
                        - i
                    )
                else:
                    tempt = set(
                        clusters_array[0][ii][0][
                            (pre < clusters_array[0][ii][0])
                            * (clusters_array[0][ii][0] < post)
                        ]
                    )
                flag = 0
            else:
                if subtraction:
                    tempt = tempt.union(
                        set(
                            clusters_array[0][ii][0][
                                (pre < clusters_array[0][ii][0])
                                * (clusters_array[0][ii][0] < post)
                            ]
                            - i
                        )
                    )
                else:
                    tempt = tempt.union(
                        set(
                            clusters_array[0][ii][0][
                                (pre < clusters_array[0][ii][0])
                                * (clusters_array[0][ii][0] < post)
                            ]
                        )
                    )
        spike_time_list.append(np.array(list(tempt)))
    return spike_time_list


def load_singing_mouse_session(data_dir, **kwargs):
    ppd = kwargs.get("psth_param_dict", None)
    psth_param_dict = parse_psth_params(ppd)
    cpu = kwargs.get("cpu", 1)
    spike_time_list = kwargs.get("spike_time_list", None)
    verbose = kwargs.get("verbose", 1)
    alignment = kwargs.get("alignment", None)
    smoothing_dict = kwargs.get("smoothing_dict", None)

    audio_data_mat = spio.loadmat(os.path.join(data_dir, "BehavioralTimings.mat"))
    start_sing_time_array = audio_data_mat["T_Motor"][:, 0]
    end_sing_time_array = audio_data_mat["T_Motor"][:, 1]
    song_length_array = end_sing_time_array - start_sing_time_array
    sensory_array = audio_data_mat["T_Sensory"]

    neural_data_mat = spio.loadmat(os.path.join(data_dir, "clusterOutput.mat"))
    clusters_array = neural_data_mat["clusters"]
    if spike_time_list is None:
        spike_time_list = []
        for ii in range(0, clusters_array[0].shape[0]):
            spike_time_list.append(clusters_array[0][ii][0])
    if alignment is None:
        alignment = start_sing_time_array
    elif type(alignment) is str and alignment == "on":
        alignment = start_sing_time_array
    elif type(alignment) is str and alignment == "off":
        alignment = end_sing_time_array
    elif type(alignment) is str and alignment == "both":
        pass
    elif type(alignment) is str:
        raise Exception("Unrecognized alignment")

    all_sing_time_psth, bin_vec = generate_PSTH(
        psth_param_dict, alignment, spike_time_list, cpu, verbose
    )

    return (
        all_sing_time_psth,
        start_sing_time_array,
        end_sing_time_array,
        song_length_array,
        sensory_array,
        bin_vec,
    )


def get_syllable_on_off_times(audio_data_mat, new=False):
    song_num = len(audio_data_mat["SyllStartStopTimes"])
    all_song_syl_on_list = []
    all_song_syl_off_list = []
    all_song_syl_length_list = []
    song_type_list = []
    all_song_start_list = []
    for ii in range(0, song_num):
        cur_syl_on_list = []
        cur_syl_off_list = []
        cur_syl_length_list = []
        syl_num = len(audio_data_mat["SyllStartStopTimes"][ii]["Ons"])  # new format
        if syl_num >= 1:
            for jj in range(0, syl_num):
                cur_syl_on_list.append(
                    audio_data_mat["SyllStartStopTimes"][ii]["Ons"][jj]
                )  # new format
                cur_syl_off_list.append(
                    audio_data_mat["SyllStartStopTimes"][ii]["Offs"][jj]
                )
                cur_syl_length_list.append(cur_syl_off_list[jj] - cur_syl_on_list[jj])
        try:
            song_type_list.append(audio_data_mat["SyllStartStopTimes"][ii]["Label"])
        except:
            song_type_list.append("motor")
        try:
            all_song_start_list.append(
                audio_data_mat["SyllStartStopTimes"][ii]["SongStart"]
            )
        except:
            all_song_start_list.append(0)
        all_song_syl_on_list.append(cur_syl_on_list)
        all_song_syl_off_list.append(cur_syl_off_list)
        all_song_syl_length_list.append(cur_syl_length_list)
    if new:
        return (
            all_song_syl_length_list,
            all_song_syl_on_list,
            all_song_syl_off_list,
            np.array(song_type_list),
            np.array(all_song_start_list),
        )
    return all_song_syl_length_list, all_song_syl_on_list, all_song_syl_off_list


def generate_PSTH(psth_param_dict, event_times, spike_time_list, cpu=1, verbose=1):
    if verbose:
        print("Using %d CPUs" % cpu)
    event_num = np.shape(event_times)[0]
    neuron_num = len(spike_time_list)
    bin_vec = np.arange(
        start=-psth_param_dict["pre_length"],
        stop=psth_param_dict["post_length"] + psth_param_dict["bin_length"],
        step=psth_param_dict["bin_length"],
    )
    bin_num = np.shape(bin_vec)[0] - 1
    psth_array = np.zeros((event_num, bin_num, neuron_num))
    for ii in range(0, neuron_num):
        for jj in range(0, event_num):
            psth_array[jj, :, ii] = np.histogram(
                spike_time_list[ii], event_times[jj] + bin_vec
            )[0]
        if verbose:
            print("Finished processing %d/%d neurons" % (ii, neuron_num - 1))
    return psth_array, bin_vec


def parse_psth_params(psth_param_dict):
    p = dict()
    p["pre_length"] = 1
    p["post_length"] = 13
    p["bin_length"] = 0.05

    if psth_param_dict == None:
        print("No psth param dict given")
        psth_param_dict = p
    for key, val in p.items():
        if key not in psth_param_dict:
            psth_param_dict[key] = val
    return psth_param_dict


def kernel(t, sigma):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-((t / sigma) ** 2) / 2)


def multiprocessing_help_func_s(
    spike_time_list,
    start_sing_time_array,
    cutoff_range,
    sample_size,
    sample_space,
    sigma,
    ii,
    jj,
):
    sp_list_neuron = np.array(spike_time_list[ii]) - start_sing_time_array[jj]
    if ((sp_list_neuron < cutoff_range) * (sp_list_neuron > -cutoff_range)).sum() == 0:
        return np.zeros(len(sample_space))
    sp_time_shift = (
        np.tile(
            sp_list_neuron[
                (sp_list_neuron < cutoff_range) * (sp_list_neuron > -cutoff_range)
            ],
            (sample_size, 1),
        )
        - sample_space[:, None]
    )
    return kernel(sp_time_shift, sigma).sum(axis=1)


def load_singing_mouse_session_smooth(data_dir, **kwargs):
    ppd = kwargs.get("psth_param_dict", None)
    psth_param_dict = parse_psth_params(ppd)
    cpu = kwargs.get("cpu", 1)
    spike_time_list = kwargs.get("spike_time_list", None)
    verbose = kwargs.get("verbose", 1)
    alignment = kwargs.get("alignment", None)
    smoothing_dict = kwargs.get("smoothing_dict", None)

    audio_data_mat = spio.loadmat(os.path.join(data_dir, "BehavioralTimings.mat"))
    start_sing_time_array = audio_data_mat["T_Motor"][:, 0]
    end_sing_time_array = audio_data_mat["T_Motor"][:, 1]
    song_length_array = end_sing_time_array - start_sing_time_array
    sensory_array = audio_data_mat["T_Sensory"]

    neural_data_mat = spio.loadmat(os.path.join(data_dir, "clusterOutput.mat"))
    clusters_array = neural_data_mat["clusters"]
    if spike_time_list is None:
        spike_time_list = []
        for ii in range(0, clusters_array[0].shape[0]):
            spike_time_list.append(clusters_array[0][ii][0])
    if alignment is None:
        alignment = start_sing_time_array
    elif type(alignment) is str and alignment == "on":
        alignment = start_sing_time_array
    elif type(alignment) is str and alignment == "off":
        alignment = end_sing_time_array
    elif type(alignment) is str and alignment == "both":
        raise Exception("Not implemented")
    elif type(alignment) is str:
        raise Exception("Unrecognized alignment")
    # sample_size=smoothing_dict.get('sample_size',1000)
    event_num = np.shape(alignment)[0]
    neuron_num = len(spike_time_list)
    sample_space = np.arange(
        start=-psth_param_dict["pre_length"],
        stop=psth_param_dict["post_length"] + psth_param_dict["bin_length"],
        step=psth_param_dict["bin_length"],
    )
    cutoff_range = smoothing_dict.get("cutoff_range", 100)
    sigma = smoothing_dict.get("sigma", 1)
    sample_size = len(sample_space)
    psth_array = np.zeros((event_num, len(sample_space), neuron_num))
    for ii in range(0, neuron_num):
        with Pool(cpu) as p:
            func = partial(
                multiprocessing_help_func_s,
                spike_time_list,
                alignment,
                cutoff_range,
                sample_size,
                sample_space,
                sigma,
                ii,
            )
            psth_array[:, :, ii] = np.array(p.map(func, np.arange(event_num)))
        if verbose:
            print("Finished processing %d/%d neurons" % (ii, neuron_num - 1))

    return (
        psth_array,
        start_sing_time_array,
        end_sing_time_array,
        song_length_array,
        sensory_array,
        sample_space,
    )
