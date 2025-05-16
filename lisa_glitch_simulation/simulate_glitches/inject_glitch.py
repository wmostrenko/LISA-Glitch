"""
Inject glitches into LISA data
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from pytdi.michelson import X2, Y2, Z2
# from ldc.utils.logging import init_logger, close_logger
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from lisainstrument.containers import ForEachMOSA
from lisainstrument import Instrument
from pytdi from Data import from_instrument
import argparse

start_time = time.time()  # timing stuff

PATH_cd = os.getcwd()
PATH_lgs = os.path.abspath(os.path.join(PATH_cd, os.pardir))  # PATH to lisa_glitch_simulation directory
PATH_io = os.path.join(os.path.abspath(os.path.join(PATH_cd, os.pardir)), 'input_output') # Maybe + "/"
PATH_tdi_out = os.path.join(PATH_lgs, 'final_tdi_outputs')


def init_cl():
    """
    initialize the command line arguments
    """

    parser = argparse.ArgumentParser()

    # FILE MANAGEMENT
    parser.add_argument(
        '--glitch_input_h5',
        type=str,
        default=None, #check if default is auto None, if so remove
        help="Glitch input h5 file path"
    )
    parser.add_argument(
        '--glitch_input_txt',
        type=str,
        default=None,
        help="Glitch input txt file path"
    )
    parser.add_argument(
        '--simulation_output_h5',
        type=str,
        default=None,
        help="Pre-TDI LISA simulation output h5 file path"
    )
    parser.add_argument(
        '--tdi_output_h5',
        type=str,
        default=None,
        help="TDI channels output h5 file path"
    )

    # LISA INSTRUMENT ARGUMENTS
    parser.add_argument(
        '--disable_noise',
        type=bool,
        default=False,
        help="Simulate LISA instruments without noise?"
    )

    args = parser.parse_args()
    # logger = init_logger(args.log, name='lisaglitch.glitch')

    return args


def init_glitch_inputs(glitch_input_txt):
    """
    initialize the input variables
    """

    glitch_inputs = np.genfromtxt(PATH_io + '/' + glitch_input_txt)

    glitch_inputs_dict = {
        'n_samples': glitch_input[1:, 1][0],
        'dt': glitch_input[1:, 2][0],
        't0': glitch_input[1:, 3][0],
        'physics_upsampling': 1.0,
        'dt_physic': glitch_input[1:, 2][0] / 1.0,
        'aafilter': None,
    }

    return glitch_inputs_dict


def simulate_lisa(glitch_file_h5_path, simulation_output_h5_path, glitch_inputs, disable_noise):
    """simulate the lisa instrument with the glitch file to be injected

    Args:
    glitch_file (string): filename and path to the glitch file from make_glitch
    glitch_inputs (dict): dictionary of inputs from the glitch .txt file from make_glitch
    noise (bool): if True add noise, False disable all noise

    Returns:
    lisainstrument simulation object
    """

    lisa_instrument = Instrument(
        size=glitch_inputs['n_samples'],
        dt=glitch_inputs['dt'],
        physics_upsampling=glitch_inputs['physics_upsampling'],
        aafilter=glitch_inputs['aafilter']
    )

    lisa_instrument.laser_asds = ForEachMOSA(0)  # Remove laser noise because doesn't work with PyTDI # investigate this
    # i.oms_isc_carrier_asds = ForEachMOSA(glitch_inputs['noise_dict']["readoutnoise"])

    if disable_nosie:
        lisa_instrument.disable_all_noises()

    lisa_instrument.write(simulation_output_h5_path)


def compute_tdi_channels(tdi_output_h5_path, t0, dt):
    """create the TDI channels X, Y, Z using PyTDI

    Args
    i (lisainstrument simulation object): the simulation of a lisa-like set-up
    channels (PyTDI michelson variables): second gen michelson variables from PyTDI
    inputs (dict): dictionary of inputs from the glitch .txt file from make_glitch
    tdi_names (list): list of the TDI channel names in same order as channels

    Returns
    dict of all constructed TDI channels
    """

    channels = [X2, Y2, Z2]
    tdi_names = ['X', 'Y', 'Z']
    tdi_dict = TimeSeriesDict()

    for i in range(len(channels)):
        channel = channels[i]
        
        data = from_instrument(tdi_output_h5_path)
        data.delay_derivative = None

        tdi_calculator = channel.build(data.args)
        tdi_data = tdi_calculator(data.measurements)

        window = tukey(tdi_data.size, alpha=0.001)
        tdi_dict[tdi_names[i]] = TimeSeries(tdi_data*window, t0=t0, dt=dt)

    tdi_dict.write(tdi_output_h5_path, overwrite=True)


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

# remove defaults here and instaed set them in the run inject_glitch alone part
# also i don't like having these function names be "main" would rather have them as file names, but perhaps check official style guide regarding that first
def main(glitch_input_h5, glitch_input_txt, simulation_output_h5, tdi_output_h5, disable_noise):
    # instead of having clean, just specifty an empty glitch file
    # should be able to specify in main whether or not we want noise (or in command line)
    # two ways to run: from calling main or from direct command line
    # tdi_start_t = time.time()

    # if command line, then set parameter variables to 
    cl_args = init_cl()

    if cl_args.tdi_output_h5 is not None:
        glitch_input_h5 = cl_args.glitch_input_h5
        glitch_input_txt = cl_args.glitch_input_txt
        simulation_output_h5 = cl_args.simulation_output_h5
        tdi_output_h5 = cl_args.tdi_output_h5
        disable_noise = cl_args.disable_noise

    # if clean:
    #     fname_in_h5 = glitch_file_h5
    #     fname_in_txt = glitch_file_txt
    #     fname_out = glitch_file_op
    #     noise = True

    # else:
    #     fname_in_h5 = glitch_file_h5
    #     fname_in_txt = glitch_file_txt
    #     fname_out = glitch_file_op
    #     noise = True

    glitch_inputs = init_glitch_inputs(glitch_input_txt)

    simulate_lisa(PATH_io + '/' + glitch_file_h5, SOMEPATH + simulation_output_h5, glitch_inputs, disable_noise)

    compute_tdi_channels(SOMEPATH + tdi_output_h5, glitch_inputs["dt"], glitch_inputs["t0"])

    # save_tdi(tdi_dict, fname_out, PATH_tdi_out)

    # tdi_end_t = time.time()
    # print("TDI Time: ")
    # print("--- %s seconds ---" % (tdi_end_t - tdi_start_t))


"""Uncomment to run inject_glitch alone"""
# if __name__ == "__main__":
#
#     main()



