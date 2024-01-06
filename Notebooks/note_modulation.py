import numpy as np
import os
import singing_mouse_aux_functions as aux_functions
from song_modulation import *
import copy
from functools import partial
from multiprocessing import Pool

alpha = 0.01  # significance level
pause_threshold = 3  # threshold multiplier for pause


def split_syl(
    all_song_syl_length_list,
    all_song_syl_on_list,
    all_song_syl_off_list,
    song_type_list,
):
    motor1 = (song_type_list == "motor") + (song_type_list == "Motor")
    auditory1 = song_type_list == "auditory"
    syl_dict = {
        "motor": (
            np.array(all_song_syl_length_list)[motor1],
            np.array(all_song_syl_on_list)[motor1],
            np.array(all_song_syl_off_list)[motor1],
        ),
        "auditory": (
            np.array(all_song_syl_length_list)[auditory1],
            np.array(all_song_syl_on_list)[auditory1],
            np.array(all_song_syl_off_list)[auditory1],
        ),
    }
    return syl_dict


def kernel(t, sigma):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-((t / sigma) ** 2) / 2)


class session_note_modulation(session_song_modulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_notes()

    def load_notes(self):
        audio_data_mat = aux_functions.loadmat(
            os.path.join(self.data_dir, "BehavioralTimings.mat")
        )
        (
            all_song_syl_length_list,
            all_song_syl_on_list,
            all_song_syl_off_list,
            song_type_list,
            all_song_start_array,
        ) = aux_functions.get_syllable_on_off_times(audio_data_mat, new=True)
        self.sp_list = aux_functions.cache_spike_counts(
            self.data_dir, pre_length=10, post_length=20, alignment=all_song_start_array
        )
        self.syl_dict = split_syl(
            all_song_syl_length_list,
            all_song_syl_on_list,
            all_song_syl_off_list,
            song_type_list,
        )
        self.song_type_list = song_type_list

    def select_note_modulation_neurons(self, condition="motor", threshold=[0, 1]):
        self.note_modulation = dict()
        self.note_modulation_p = dict()
        self.note_modulation_r = dict()
        (
            all_song_syl_length_list,
            all_song_syl_on_list,
            all_song_syl_off_list,
        ) = self.syl_dict[condition]
        all_syl_on_list = np.concatenate(all_song_syl_on_list)
        all_syl_off_list = np.concatenate(all_song_syl_off_list)
        all_syl_on_list = np.concatenate(all_song_syl_on_list)
        all_syl_off_list = np.concatenate(all_song_syl_off_list)
        neuron_selection = neuron_selection_for_note_tuned_neurons(
            all_syl_on_list,
            all_syl_off_list,
            self.sp_list,
            None,
            self.neuron_num,
            threshold=threshold,
        )
        self.note_modulation[condition] = neuron_selection[0]
        self.note_modulation_p[condition] = neuron_selection[1]
        self.note_modulation_r[condition] = neuron_selection[2]


def circ_test(return_list, duration_list, weights=True, return_value=False):
    empirical_ob = np.concatenate(return_list) * np.pi * 2
    if len(empirical_ob) == 0:
        print("Warning: Func: circ_test\nEmpty list encountered")
        if return_value:
            return 1, 0, 0
        else:
            return 1
    if weights == True:
        print(len(empirical_ob))
        w = np.concatenate(
            [
                np.ones(len(listi)) / duration_list[i]
                for i, listi in enumerate(return_list)
            ]
        )
        y = np.sum(np.sin(empirical_ob) * w) / w.sum()
        x = np.sum(np.cos(empirical_ob) * w) / w.sum()
    else:
        y = (np.sin(empirical_ob)).mean()
        x = (np.cos(empirical_ob)).mean()
    r = np.sqrt(x**2 + y**2)
    n = len(empirical_ob)
    z = n * (r**2)
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 * (1 - r**2))) - (1 + 2 * n))
    if return_value:
        return (
            pval,
            r,
            np.sqrt(
                (np.sin(empirical_ob).std() * y / r) ** 2
                + (np.cos(empirical_ob).std() * x / r) ** 2
            )
            / np.sqrt(n),
        )
    else:
        return pval


def neuron_selection_for_note_tuned_neurons(
    all_syl_on_list,
    all_syl_off_list,
    sp_list,
    all_sing_time_psth,
    neuron_num,
    threshold,
    alpha=0.01,
):
    p_list = []
    strength_list = []
    for neuron_id in range(neuron_num):
        return_list, _, _, duration_list = transform_to_relative_time2_list_latency(
            all_syl_on_list,
            all_syl_off_list,
            sp_list[neuron_id],
            threshold=threshold,
            order=None,
            latency=0,
        )
        p = circ_test(return_list, duration_list, False, True)
        p_list.append(p[0])
        strength_list.append(p[1])
    p_list = np.array(p_list)
    strength_list = np.array(strength_list)
    neuron_selection = np.where(p_list < alpha)[0]
    return neuron_selection, p_list, strength_list


def transform_to_relative_time2_list_latency(
    all_syl_on_list,
    all_syl_off_list,
    neuron_sp_list,
    threshold=False,
    order=None,
    latency=0.8,
):
    def interpolate_to_relative_time(st, en, data_list):
        return (data_list - st) / (en - st)

    syl_num = len(all_syl_on_list)
    return_list = []
    order_list = []
    off_list = []
    history = []  # -1 off and +1 on
    duration_list = []
    if type(threshold) is bool:
        if threshold:
            threshold = [0.1, 1e10]
        else:
            threshold = [0.0, 1e10]
    for i in range(syl_num - 1):
        if (
            all_syl_on_list[i + 1] - all_syl_on_list[i] >= threshold[0]
            and all_syl_on_list[i + 1] - all_syl_on_list[i] <= threshold[1]
        ) and all_syl_on_list[i + 1] - all_syl_on_list[i] < pause_threshold * (
            all_syl_off_list[i] - all_syl_on_list[i]
        ):
            tempt_list = interpolate_to_relative_time(
                all_syl_on_list[i], all_syl_on_list[i + 1], neuron_sp_list
            )
            tempt_list = tempt_list[
                (tempt_list >= 0 - latency) * (tempt_list <= 1 + latency)
            ]
            if True:
                if i == 0:
                    previous_off = 0
                else:
                    previous_off = interpolate_to_relative_time(
                        all_syl_on_list[i],
                        all_syl_on_list[i + 1],
                        all_syl_off_list[i - 1],
                    )
                future_on = interpolate_to_relative_time(
                    all_syl_on_list[i], all_syl_on_list[i + 1], all_syl_on_list[i + 1]
                )
                history.append([previous_off, future_on])
                return_list.append(tempt_list)
                off_list.append(
                    interpolate_to_relative_time(
                        all_syl_on_list[i], all_syl_on_list[i + 1], all_syl_off_list[i]
                    )
                )
                duration_list.append(all_syl_on_list[i + 1] - all_syl_on_list[i])
                if not order is None:
                    order_list.append(order[i])
    if not order is None:
        return (
            np.array(return_list)[np.argsort(order_list)],
            np.array(off_list)[np.argsort(order_list)],
            np.array(history)[np.argsort(order_list)],
            np.array(duration_list)[np.argsort(order_list)],
        )
    else:
        return (
            return_list,
            np.array(off_list),
            np.array(history),
            np.array(duration_list),
        )


def transform_to_absolute_time_list_help_func(
    all_syl_on_list,
    all_syl_off_list,
    neuron_sp_list,
    threshold=True,
    order=None,
    align="on",
    latency=0,
    threshold_history=True,
    return_transformed_syl_time=False,
):
    syl_num = len(all_syl_on_list)
    return_list = []
    order_list = []
    on_list = []
    off_list = []
    history = []  # -1 off and +1 on
    syl_transformed_list = []
    if type(threshold) is bool:
        if threshold:
            threshold = [0.1, 1e10]
        else:
            threshold = [0.0, 1e10]
    for i in range(syl_num - 1):
        if (
            (
                all_syl_on_list[i + 1] - all_syl_on_list[i] >= threshold[0]
                and all_syl_on_list[i + 1] - all_syl_on_list[i] <= threshold[1]
            )
            and all_syl_on_list[i + 1] - all_syl_off_list[i]
            < (all_syl_off_list[i] - all_syl_on_list[i])
            and (
                not threshold_history
                or (
                    i != 0
                    and all_syl_on_list[i] - all_syl_off_list[i - 1]
                    < (all_syl_off_list[i] - all_syl_on_list[i])
                )
            )
        ):
            tempt_list = neuron_sp_list[
                (neuron_sp_list >= all_syl_on_list[i] - latency)
                * (neuron_sp_list <= all_syl_on_list[i + 1] + latency)
            ]
            assert align in ["on", "off"]
            if align == "on":
                alignment = all_syl_on_list[i]
            else:
                alignment = all_syl_off_list[i]
            tempt_list = tempt_list - alignment
            if i == 0:
                previous_off = 0
            else:
                previous_off = all_syl_off_list[i - 1] - alignment
            future_on = all_syl_on_list[i + 1] - alignment
            history.append([previous_off, future_on])
            return_list.append(tempt_list)
            if order is not None:
                order_list.append(order[i])
            if align == "on":
                on_list.append(0)
                off_list.append(all_syl_off_list[i] - all_syl_on_list[i])
            else:
                on_list.append(-all_syl_off_list[i] + all_syl_on_list[i])
                off_list.append(0)
            if return_transformed_syl_time:
                syl_transformed_list.append(
                    [all_syl_on_list - alignment, all_syl_off_list - alignment]
                )
    if order is not None:
        if return_transformed_syl_time:
            return (
                np.array(return_list)[np.argsort(order_list)],
                np.array(on_list)[np.argsort(order_list)],
                np.array(off_list)[np.argsort(order_list)],
                np.array(history)[np.argsort(order_list)],
                np.array(syl_transformed_list)[np.argsort(order_list)],
            )
        else:
            return (
                np.array(return_list)[np.argsort(order_list)],
                np.array(on_list)[np.argsort(order_list)],
                np.array(off_list)[np.argsort(order_list)],
                np.array(history)[np.argsort(order_list)],
            )
    else:
        return return_list, on_list, off_list, history


def rank_order(
    all_song_syl_on_list,
    all_song_syl_off_list,
    start_sing_time_array,
    song_length_array,
    rank="duration",
):
    if rank == "time":
        syl_on_list = copy.deepcopy(all_song_syl_on_list)
        for i in range(len(start_sing_time_array)):
            syl_on_list[i] = np.array(syl_on_list[i]) - start_sing_time_array[i]
        syl_on_list = np.concatenate(syl_on_list)
        return syl_on_list
    elif rank == "duration":
        syl_on_list = np.concatenate(all_song_syl_on_list)
        syl_off_list = np.concatenate(all_song_syl_off_list)
        syl_duration = -syl_on_list + syl_off_list
        return syl_duration
    elif rank == "relative time":
        syl_on_list = copy.deepcopy(all_song_syl_on_list)
        for i in range(len(start_sing_time_array)):
            syl_on_list[i] = (
                np.array(syl_on_list[i]) - start_sing_time_array[i]
            ) / song_length_array[i]
        syl_on_list = np.concatenate(syl_on_list)
        return syl_on_list


def raster_plot_concate_spike_list(spike_list, threshold=[-0.1, 0.5]):
    x = []
    y = []
    for i in range(len(spike_list)):
        for_plot = np.array(spike_list[i])
        for_plot = for_plot[
            ((for_plot > threshold[0]) * (for_plot < threshold[1])).astype("bool")
        ]
        x.extend(list(for_plot))
        y.extend(list(np.ones(len(for_plot)) * i))
    return np.array(x), np.array(y)


def window_average_1d(x, window, threshold=None):
    if threshold is not None:
        newx = x[(x > threshold[0]) * (x < threshold[1])]
    else:
        newx = x
    return_list = copy.deepcopy(newx)
    tempt = np.convolve(x, np.ones(window), "same") / window
    return_list[window // 2 : -window // 2] = tempt[window // 2 : -window // 2]
    return_list[: window // 2] = np.ones(window // 2) * tempt[window // 2]
    return_list[-window // 2 :] = np.ones(window // 2) * tempt[-window // 2]
    return return_list


def delay_multiprocessing_help(
    all_syl_on_list, all_syl_off_list, sp_list_n, threshold, delay
):
    return_list, _, _, duration_list = transform_to_relative_time2_list_latency(
        all_syl_on_list,
        all_syl_off_list,
        sp_list_n + delay,
        threshold=threshold,
        order=None,
        latency=0,
    )
    _, r, dr = circ_test(return_list, duration_list, False, True)
    return r / dr


def find_delay(
    delay_set, all_syl_on_list, all_syl_off_list, sp_list_n, threshold, cpu=16
):
    delay_multiprocessing_hep2 = partial(
        delay_multiprocessing_help,
        all_syl_on_list,
        all_syl_off_list,
        sp_list_n,
        threshold,
    )
    with Pool(cpu) as p:
        results = p.map(delay_multiprocessing_hep2, delay_set)
    return delay_set[np.argmax(results)], results
