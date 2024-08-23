"""
Inject glitches into LISA data
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from pytdi.michelson import X2, Y2, Z2
from ldc.utils.logging import init_logger, close_logger
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from lisainstrument.containers import ForEachMOSA
from lisainstrument import Instrument
from pytdi import Data

start_time = time.time()  # timing stuff

PATH_cd = os.getcwd()
PATH_lgs = os.path.abspath(os.path.join(PATH_cd, os.pardir))  # PATH to lisa_glitch_simulation directory
PATH_io = os.path.join(os.path.abspath(os.path.join(PATH_cd, os.pardir)), 'input_output')
PATH_tdi_out = os.path.join(PATH_lgs, 'final_tdi_outputs')

TDI_VAR = [X2, Y2, Z2]
TDI_NAMES = ['X', 'Y', 'Z']


def init_cl():
    """
    initialize the command line arguments
    """

    import argparse
    parser = argparse.ArgumentParser()
    # Main arguments
    parser.add_argument('--path-input', type=str, default=PATH_io, help="Path to input glitch files")
    parser.add_argument('--path-output', type=str, default=PATH_cd, help="Path to save output tdi files")
    parser.add_argument('--glitch-h5-mg-output', type=str, default="", help="Glitch output h5 file")
    parser.add_argument('--glitch-txt-mg-output', type=str, default="", help="Glitch output txt file")
    parser.add_argument('--tdi-output-file', type=str, default="", help="Glitch output h5 file")
    parser.add_argument('--no-glitches', type=bool, default=False, help="Want Glitches?")
    parser.add_argument('--noise', type=bool, default=True, help="Want noise?")
    parser.add_argument('-l', '--log', default="", help="Log file")
    args = parser.parse_args()
    logger = init_logger(args.log, name='lisaglitch.glitch')

    return args


def init_inputs(glitch_info, old_file=False):
    """
    initialize the input variables
    """

    if not old_file:
        g_info = np.genfromtxt(PATH_io + '/' + glitch_info)
        n_samples = g_info[1:, 1][0]
        g_dt = g_info[1:, 2][0]
        g_t0 = g_info[1:, 4][0]
        g_physics_upsampling = g_info[1:, 3][0]
        dt_physic = g_dt / g_physics_upsampling
    else:
        g_info = np.genfromtxt(PATH_io + '/' + glitch_info)
        n_samples = g_info[1:, 1][0]
        g_dt = g_info[1:, 2][0]
        g_t0 = g_info[1:, 3][0]
        g_physics_upsampling = 1.0
        dt_physic = g_dt / g_physics_upsampling

    central_freq = 2.816E14
    aafilter = None

    d = {"backlinknoise": 3e-12, "accnoise": 2.4e-15, "readoutnoise": 6.35e-12}

    g_inputs = {'n_samples': n_samples, 'dt': g_dt, 't0': g_t0, 'physics_upsampling': g_physics_upsampling,
                'dt_physic': dt_physic, 'central_freq': central_freq, 'aafilter': aafilter,
                'noise_dict': d}

    return g_inputs


def simulate_lisa(glitch_file, glitch_inputs, noise=True, clean=False):
    """simulate the lisa instrument with the glitch file to be injected

    Args:
    glitch_file (string): filename and path to the glitch file from make_glitch
    glitch_inputs (dict): dictionary of inputs from the glitch .txt file from make_glitch
    noise (bool): if True add noise, False disable all noise

    Returns:
    lisainstrument simulation object
    """

    # Inject Glitches
    if clean:
        print('here')
        i = Instrument(physics_upsampling=glitch_inputs['physics_upsampling'], aafilter=glitch_inputs['aafilter'],
                       size=glitch_inputs['n_samples'], dt=glitch_inputs['dt'],
                       central_freq=glitch_inputs['central_freq'],
                       backlink_asds=glitch_inputs['noise_dict']["backlinknoise"],
                       testmass_asds=glitch_inputs['noise_dict']["accnoise"])
        if noise:
            i.oms_isc_carrier_asds = ForEachMOSA(glitch_inputs['noise_dict']["readoutnoise"])

            i.laser_asds = ForEachMOSA(0)  # Remove laser noise because doesn't work with PyTDI

            i.disable_clock_noises()
            i.modulation_asds = ForEachMOSA(0)
            i.disable_ranging_noises()
            i.disable_dopplers()
        else:
            i.disable_all_noises()

        i.simulate()  # Run simulator

        return i

    else:
        i = Instrument(physics_upsampling=glitch_inputs['physics_upsampling'], aafilter=glitch_inputs['aafilter'],
                       size=glitch_inputs['n_samples'], dt=glitch_inputs['dt'],
                       central_freq=glitch_inputs['central_freq'],
                       backlink_asds=glitch_inputs['noise_dict']["backlinknoise"],
                       testmass_asds=glitch_inputs['noise_dict']["accnoise"], glitches=glitch_file)

        if noise:
            i.oms_isc_carrier_asds = ForEachMOSA(glitch_inputs['noise_dict']["readoutnoise"])

            i.laser_asds = ForEachMOSA(0)  # Remove laser noise because doesn't work with PyTDI
        else:
            i.disable_all_noises()

        i.simulate()  # Run simulator

        return i


def tdi_channels(i, channels, inputs, tdi_names):
    """create the TDI channels X, Y, Z using PyTDI

    Args
    i (lisainstrument simulation object): the simulation of a lisa-like set-up
    channels (PyTDI michelson variables): second gen michelson variables from PyTDI
    inputs (dict): dictionary of inputs from the glitch .txt file from make_glitch
    tdi_names (list): list of the TDI channel names in same order as channels

    Returns
    dict of all constructed TDI channels
    """

    tdis = TimeSeriesDict()
    for j in range(len(channels)):
        ch = channels[j]

        data = Data.from_instrument(i)
        data.delay_derivative = None

        built = ch.build(delays=data.delays, fs=data.fs)

        tdi_data = built(data.measurements)/inputs['central_freq']

        # Window out the tdi channels - tukey window
        win = tukey(tdi_data.size, alpha=0.001)
        tdis[tdi_names[j]] = TimeSeries(tdi_data*win, t0=inputs['t0'], dt=inputs['dt'])

    return tdis


def plot_tdi(tdi, tdi_name, xlims=None, ylims=None):

    times_arr = np.arange(0, 172800, 0.25)

    plt.figure(figsize=(10, 8))
    plt.plot(times_arr, tdi, label=f'raw TDI {tdi_name}')

    plt.title(f'TDI {tdi_name}')
    plt.xlabel('times [s]')
    plt.xlabel('amplitude')

    plt.legend()
    plt.show()


def save_tdi(tdi, output_fname, output_path):

    tdi.write(f'{output_path}/{output_fname}', overwrite=True)


def main(glitch_file_h5=None, glitch_file_txt=None, glitch_file_op=None, clean=False):

    tdi_start_t = time.time()

    if clean:
        fname_in_h5 = glitch_file_h5
        fname_in_txt = glitch_file_txt
        fname_out = glitch_file_op
        noise = True

    else:
        fname_in_h5 = glitch_file_h5
        fname_in_txt = glitch_file_txt
        fname_out = glitch_file_op
        noise = True

    inputs = init_inputs(fname_in_txt, old_file=True)

    sim = simulate_lisa(PATH_io + '/' + fname_in_h5, inputs, noise=noise, clean=clean)

    tdi_dict = tdi_channels(sim, TDI_VAR, inputs, TDI_NAMES)

    save_tdi(tdi_dict, fname_out, PATH_tdi_out)

    tdi_end_t = time.time()
    print("TDI Time: ")
    print("--- %s seconds ---" % (tdi_end_t - tdi_start_t))


"""Uncomment to run inject_glitch alone"""
# if __name__ == "__main__":
#
#     main()



