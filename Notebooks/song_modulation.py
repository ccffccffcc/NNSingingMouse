import numpy as np
import os
from scipy.stats import ttest_rel
import scipy.io as spio
import singing_mouse_aux_functions as aux_functions
import itertools
import jenkspy
from scipy.interpolate import interp1d
from scipy.stats.stats import pearsonr

alpha = 0.01  # significance level


def make_interp1d_psth(all_sing_time_psth_s, bin_vec_s, cluster_list, return_sem=False):
    song_num = all_sing_time_psth_s.shape[0]
    neuron_num = all_sing_time_psth_s.shape[2]
    inter_func = []
    inter_func_std = []
    for neuron_i in range(neuron_num):
        tempt_list = []
        tempt_list2 = []
        for cluster_i in cluster_list:
            tempt_list.append(
                interp1d(
                    bin_vec_s, all_sing_time_psth_s[cluster_i, :, neuron_i].mean(axis=0)
                )
            )
            tempt_list2.append(
                interp1d(
                    bin_vec_s,
                    all_sing_time_psth_s[cluster_i, :, neuron_i].std(axis=0)
                    / np.sqrt(len(cluster_i) - 1),
                )
            )
        inter_func.append(tempt_list)
        inter_func_std.append(tempt_list2)
    if return_sem:
        return inter_func, inter_func_std
    return inter_func


# check if overlap with song
def create_control(st, length, start_sing_time_array, end_sing_time_array):
    # first check if st is in a song
    flag = 0
    for i in range(len(start_sing_time_array)):
        if st > start_sing_time_array[i] and st < end_sing_time_array[i]:
            flag = 1
            break
    if flag:
        st = end_sing_time_array[i]
    break_points = []
    future = start_sing_time_array - st
    index = np.where(future >= 0)[0]
    if len(index) > 0:
        future = future[future >= 0]
        index = index[future.argmin()]
        while min(future) < length:
            break_points.append([st, st + min(future)])
            st = end_sing_time_array[index]
            length -= min(future)
            future = start_sing_time_array - st
            index = np.where(future >= 0)[0]
            future = future[future >= 0]
            index = index[future.argmin()]
    break_points.append([st, st + length])
    return break_points


class session_song_modulation:
    def __init__(self, data_dir, p, p2, smoothing_dict=None, cpu=16):
        self.data_dir = data_dir
        self.p = p
        self.p2 = p2

        # load data
        self.sp_list = sp_list = aux_functions.cache_spike_counts(
            data_dir, pre_length=55, post_length=120
        )  # cache spike counts

        (
            self.all_sing_time_psth,
            self.start_sing_time_array,
            self.end_sing_time_array,
            self.song_length_array,
            self.sensory_array,
            self.bin_vec,
        ) = aux_functions.load_singing_mouse_session(
            data_dir, psth_param_dict=p, cpu=32, verbose=0, spike_time_list=sp_list
        )

        self.neuron_num = self.all_sing_time_psth.shape[2]
        self.song_num = self.all_sing_time_psth.shape[0]
        audio_data_mat = spio.loadmat(os.path.join(data_dir, "BehavioralTimings.mat"))

        # Classify spont and respond songs
        self.spont_song_logical = np.any(np.isnan(self.sensory_array), axis=1)
        self.spont_song_ind = np.where(self.spont_song_logical)[0]
        self.neuron_ind_list = np.arange(self.neuron_num)

        self.song_modulated_neurons()

        # load smoothed psth
        if smoothing_dict is not None:
            self.smoothing_dict = smoothing_dict
            (
                self.all_sing_time_psth_s,
                _,
                _,
                _,
                _,
                self.bin_vec_s,
            ) = aux_functions.load_singing_mouse_session_smooth(
                data_dir,
                psth_param_dict=p2,
                cpu=cpu,
                verbose=0,
                spike_time_list=None,
                smoothing_dict=smoothing_dict,
            )
            self.cal_FR_within_song()

    def song_modulated_neurons(self, c_length=60):
        st = 1  # pre and post length
        sp_list_a_on = aux_functions.cache_spike_counts(
            self.data_dir,
            pre_length=st,
            post_length=self.song_length_array.min(),
            alignment="on",
            subtraction=True,
        )
        sp_list_a_off = aux_functions.cache_spike_counts(
            self.data_dir,
            pre_length=self.song_length_array.min(),
            post_length=st,
            alignment="off",
            subtraction=True,
        )
        cached_spike_counts = aux_functions.cache_spike_counts(
            self.data_dir, pre_length=60, post_length=150
        )
        new_psth = np.zeros([2, self.neuron_num])
        new_psth_2 = np.zeros([self.song_num, 2, self.neuron_num])
        width = 2  # pre and post length for test
        st = -width
        end = min(self.song_length_array)
        # align with onset
        time = 0
        for song_i in range(self.song_num):
            rst = self.start_sing_time_array[song_i] + st
            rend = self.start_sing_time_array[song_i] + end
            time += rend - rst
            for neuron_i in range(self.neuron_num):
                sp_list_neruon = cached_spike_counts[neuron_i]
                n_sp = ((sp_list_neruon > rst) * (sp_list_neruon < rend)).sum()
                new_psth[0, neuron_i] += n_sp
                new_psth_2[song_i, 0, neuron_i] = n_sp / (rend - rst)
        new_psth[0, :] /= time
        # align with offset
        time = 0
        st = -min(self.song_length_array)
        end = width
        for song_i in range(self.song_num):
            rst = self.end_sing_time_array[song_i] + st
            rend = self.end_sing_time_array[song_i] + end
            time += rend - rst
            for neuron_i in range(self.neuron_num):
                sp_list_neruon = cached_spike_counts[neuron_i]
                n_sp = ((sp_list_neruon > rst) * (sp_list_neruon < rend)).sum()
                new_psth[1, neuron_i] += n_sp
                new_psth_2[song_i, 1, neuron_i] = n_sp / (rend - rst)
        new_psth[1, :] /= time
        # now obtain control
        occ_len = 10
        time = 0
        control_psth = np.zeros(self.neuron_num)
        control_psth2 = np.zeros([self.song_num, self.neuron_num])
        for song_i in range(self.song_num):
            time += c_length
            st_end_list = create_control(
                self.end_sing_time_array[song_i] + occ_len,
                c_length,
                self.start_sing_time_array - occ_len,
                self.end_sing_time_array + occ_len,
            )
            for neuron_i in range(self.neuron_num):
                sp_list_neruon = cached_spike_counts[neuron_i]
                n_sp = 0
                for rst, rend in st_end_list:
                    n_sp += ((sp_list_neruon > rst) * (sp_list_neruon < rend)).sum()
                control_psth[neuron_i] += n_sp
                control_psth2[song_i, neuron_i] = n_sp / c_length
        control_psth /= time
        self.FR_control = control_psth
        # perform t-test
        p2_array = np.zeros([self.neuron_num, 2])
        t2_array = np.zeros([self.neuron_num, 2])
        self.modulation_within_song = new_psth_2
        self.control_outside_song = control_psth2
        for neuron_i in range(self.neuron_num):
            t2_array[neuron_i, 0], p2_array[neuron_i, 0] = ttest_rel(
                new_psth_2[:, 0, neuron_i], control_psth2[:, neuron_i]
            )
            t2_array[neuron_i, 1], p2_array[neuron_i, 1] = ttest_rel(
                new_psth_2[:, 1, neuron_i], control_psth2[:, neuron_i]
            )
        self.modulation_p = p2_array
        self.modulation_t = t2_array
        self.neuron_modulation_list = np.where(p2_array.min(axis=1) * 2 <= alpha)[0]

    def cal_FR_within_song(self):
        cached_spike_counts = aux_functions.cache_spike_counts(
            self.data_dir, pre_length=60, post_length=100
        )
        FR_within_song = []
        for song_i in range(self.song_num):
            FR_within_song_perneuron = []
            for neuron_i in range(self.neuron_num):
                sp_list_neruon = cached_spike_counts[neuron_i]
                FR_within_song_perneuron.append(
                    (
                        ((sp_list_neruon) > self.start_sing_time_array[song_i])
                        * (sp_list_neruon < self.end_sing_time_array[song_i])
                    ).sum()
                )
            FR_within_song.append(FR_within_song_perneuron)
        self.FR_within_song = (
            np.array(FR_within_song).sum(axis=0) / self.song_length_array.sum()
        )


def jenkspy_clustering(song_length_array, n_min=4):
    def check_centers(labels):
        n = 0
        for i in range(0, labels.max() + 1):
            if (labels == i).sum() >= n_min:
                n = n + 1
        return n

    n_list = []
    label_list = []
    for j in range(2, 8):
        breaks = jenkspy.jenks_breaks(song_length_array, j)
        labels = np.zeros(len(song_length_array)).astype("int")
        breaks[0] -= 1
        for i in range(len(breaks) - 1):
            labels[
                (song_length_array > breaks[i]) * (song_length_array <= breaks[i + 1])
            ] = i
        n = check_centers(labels)
        n_list.append(n)
        label_list.append(labels)
    n_list = np.array(n_list) - np.arange(len(n_list)) * 1e-15
    return label_list[np.argmax(n_list)], np.max(n_list)


def combine_selection(con1, con2):
    selection = set(list(con1)).intersection(set(list(con2)))
    return np.sort(list(selection))


def extract_spike_single(neuron_spike, start_time, end_time):
    return neuron_spike[(neuron_spike > start_time) * (neuron_spike < end_time)]


def extract_spike_for_song(spike, start_time, end_time, end_latency=0, latency_start=0):
    neuron_num = len(spike)
    song_num = len(start_time)
    assert len(start_time) == len(end_time)
    neuron_song_sp_list = []
    for i in range(neuron_num):
        song_sp_list = []
        neuron_spike = spike[i]
        for j in range(song_num):
            song_sp_list.append(
                extract_spike_single(
                    neuron_spike,
                    start_time[j] - latency_start,
                    end_time[j] + end_latency,
                )
                - start_time[j]
                + latency_start
            )
        neuron_song_sp_list.append(song_sp_list)
    return neuron_song_sp_list


def cluster_spike_single(neuron_list, cluster):
    return np.concatenate(neuron_list[cluster])


def scale_single(neuron_sp_cluster_list, scale, weight, bins):
    for_bin = neuron_sp_cluster_list / scale
    hist, new_bin = np.histogram(for_bin, bins, weights=np.ones(len(for_bin)) * weight)
    return new_bin, hist


def scale_factor_analysis_single(for_scaling, bins1, for_compare, bins2, scale):
    if bins1[-1] / scale <= bins2[-1]:
        hist = for_scaling(bins2 * scale)
        hist2 = for_compare(bins2)
    else:
        hist2 = for_compare(bins1 / scale)
        hist = for_scaling(bins1)
    return hist, hist2


def scale_factor_analysis_single_overlap(for_scaling, bins1, for_compare, bins2, scale):
    if bins1[-1] / scale > bins2[-1]:
        hist = for_scaling(bins2 * scale)
        hist2 = for_compare(bins2)
    else:
        hist2 = for_compare(bins1 / scale)
        hist = for_scaling(bins1)
    return hist, hist2


class warping_analysis:
    def __init__(self, session, FR_t=1):
        self.session_song_modulation = session
        FR_condition = np.where(
            (self.session_song_modulation.FR_within_song > FR_t)
            + (self.session_song_modulation.FR_control > FR_t)
        )[0]
        self.neuron_modulation_list = combine_selection(
            self.session_song_modulation.neuron_modulation_list, FR_condition
        )

    def clustering(self, verbose=1):
        sp_list = self.session_song_modulation.sp_list
        song_length_array = self.session_song_modulation.song_length_array
        start_sing_time_array = self.session_song_modulation.start_sing_time_array
        end_sing_time_array = self.session_song_modulation.end_sing_time_array
        neuron_modulation_list = self.session_song_modulation.neuron_modulation_list
        data_dir = self.session_song_modulation.data_dir

        self.cluster_label, self.n = jenkspy_clustering(song_length_array, n_min=4)
        self.n = int(self.n)

        cluster_list = []
        for i in range(20):
            if (self.cluster_label == i).sum() >= 4:
                cluster_list.append(np.where(self.cluster_label == i)[0])
        cluster_mean = []
        for i, cluster in enumerate(cluster_list):
            cluster_mean.append(np.mean(song_length_array[cluster]))
            if verbose:
                print(
                    "Cluster %d: mean song leangth %.1f s, std %.1f s"
                    % (
                        i,
                        np.mean(song_length_array[cluster]),
                        np.std(song_length_array[cluster])
                        * np.sqrt(len(cluster) / (len(cluster) - 1)),
                    )
                )
        reindex = np.argsort(cluster_mean)

        self.cluster_mean = np.array(cluster_mean)[reindex]
        self.cluster_list = np.array(cluster_list)[reindex]

    def warping(self, **kwargs):
        verbose = kwargs.get("verbose", 0)
        self.clustering(verbose)
        self.metrics = kwargs["metrics"]
        self.low_lim, self.high_lim = kwargs["low_lim"], kwargs["high_lim"]
        self.bin_num = kwargs["bin_num"]
        self.inter_func = make_interp1d_psth(
            self.session_song_modulation.all_sing_time_psth_s,
            self.session_song_modulation.bin_vec_s,
            self.cluster_list,
        )
        self.results = []
        if len(self.cluster_list) <= 1:
            print("No enough clusters for" + self.session_song_modulation.data_dir)
            return self.results
        for neuron_index in self.neuron_modulation_list:
            self.results.append(self.warping_single_neuron(neuron_index))
        return self.results

    def warping_single_neuron(self, neuron_index, **kwargs):
        cluster_list_FR = self.inter_func[neuron_index]
        epsilon = 1e-5
        meta_loss = []
        truth_l = []
        optimal_s = []
        range_list = []
        metrics = self.metrics
        for i, j in list(itertools.combinations(np.arange(len(self.cluster_list)), 2)):
            loss_l = []
            r_g = (self.low_lim, self.high_lim)
            range_list.append(np.arange(r_g[0], r_g[1], 0.01))
            for sc in np.arange(r_g[0], r_g[1], 0.01):
                if metrics == "MSE":
                    p1, p2 = scale_factor_analysis_single(
                        cluster_list_FR[i],
                        np.linspace(0, self.cluster_mean[i], self.bin_num),
                        cluster_list_FR[j],
                        np.linspace(0, self.cluster_mean[j], self.bin_num),
                        sc,
                    )
                    p1 = p1 / (p1.sum() + epsilon)
                    p2 = p2 / (p2.sum() + epsilon)
                    loss_l.append(np.mean((p1 - p2) ** 2))
                elif metrics == "correlation":
                    p1, p2 = scale_factor_analysis_single(
                        cluster_list_FR[i],
                        np.linspace(0, self.cluster_mean[i], self.bin_num),
                        cluster_list_FR[j],
                        np.linspace(0, self.cluster_mean[j], self.bin_num),
                        sc,
                    )
                    loss_l.append(-pearsonr(p1, p2)[0])
                else:
                    raise Exception("Metric not supported")
            meta_loss.append(loss_l)
            optimal_s.append(np.arange(r_g[0], r_g[1], 0.01)[np.argmin(loss_l)])
            truth_l.append((self.cluster_mean[i]) / (self.cluster_mean[j]))
        return meta_loss, range_list, np.array(optimal_s), np.array(truth_l)


def make_interp1d_psth_single(all_sing_time_psth_s, bin_vec_s):
    song_num = all_sing_time_psth_s.shape[0]
    neuron_num = all_sing_time_psth_s.shape[2]
    inter_func = []
    for neuron_i in range(neuron_num):
        tmept_list = []
        for song_i in range(song_num):
            tmept_list.append(
                interp1d(bin_vec_s, all_sing_time_psth_s[song_i, :, neuron_i])
            )
        inter_func.append(tmept_list)
    return inter_func
