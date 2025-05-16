import os
import numpy as np
from scipy.signal.windows import tukey
from pytdi.michelson import X2, Y2, Z2
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from lisainstrument.containers import ForEachMOSA
from lisainstrument import Instrument
from pytdi import Data
import argparse


PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_glitch_config = os.path.join(PATH_bethLISA, "dist/glitch_config/")
PATH_glitch_data = os.path.join(PATH_bethLISA, "dist/glitch_data/")
PATH_simulation_data = os.path.join(PATH_bethLISA, "dist/simulation_data/")
PATH_tdi_data = os.path.join(PATH_bethLISA, "dist/tdi_data/")


def init_cl():
    """Initialize commandline arguments and return Namespace object with all
    given commandline arguments.
    """

    parser = argparse.ArgumentParser()

    # FILE MANAGEMENT
    parser.add_argument(
        "--glitch_input_h5",
        type=str,
        help="Glitch input h5 file path",
    )
    parser.add_argument(
        "--glitch_input_txt",
        type=str,
        help="Glitch input txt file path"
    )
    parser.add_argument(
        "--simulation_output_h5",
        type=str,
        help="Pre-TDI LISA simulation output h5 file path",
    )
    parser.add_argument(
        "--tdi_output_h5",
        type=str,
        help="TDI channels output h5 file path",
    )

    # LISA INSTRUMENT ARGUMENTS
    parser.add_argument(
        "--disable_noise",
        type=bool,
        default=False,
        help="Simulate LISA instruments without noise?",
    )

    return parser.parse_args()


def init_glitch_inputs(glitch_input_txt_path):
    """Create and return a dictionary containing all needed inputs from a
    glitch input file.

    Arguments:
    glitch_input_txt_path -- path to glitch input .txt file
    """

    glitch_inputs = np.genfromtxt(glitch_input_txt_path)

    glitch_inputs_dict = {
        "n_samples": glitch_inputs[1:, 1][0],
        "dt": glitch_inputs[1:, 2][0],
        "t0": glitch_inputs[1:, 3][0],
        "physics_upsampling": 1.0,
        "dt_physic": glitch_inputs[1:, 2][0] / 1.0,
        "aafilter": None,
    }

    return glitch_inputs_dict


def simulate_lisa(
    glitch_input_h5_path, simulation_output_h5_path, glitch_inputs,
    disable_noise
):
    """Simulate LISA and write output data to file.

    Arguments:
    glitch_input_h5_path -- path to glitch input .h5 file
    simulation_output_h5_path -- path to output .h5 file for LISA simulation
    outputs
    glitch_inputs -- glitch input data from .txt file as a dictionary
    disable_noise -- boolean describing if noise should be disabled
    """

    # CREATE LISA INSTRUMENT OBJECT
    lisa_instrument = Instrument(
        size=glitch_inputs["n_samples"],
        dt=glitch_inputs["dt"],
        physics_upsampling=glitch_inputs["physics_upsampling"],
        aafilter=glitch_inputs["aafilter"],
        glitches=glitch_input_h5_path,
    )

    lisa_instrument.laser_asds = ForEachMOSA(0)

    if disable_noise:
        lisa_instrument.disable_all_noises()

    # SIMULATE LISA AND SAVE RESULTS TO FILE
    lisa_instrument.write(simulation_output_h5_path)


def compute_and_save_tdi_channels(
    simulation_input_h5_path, tdi_output_h5_path, t0, dt
):
    """Compute tdi channels from LISA simulation output data and save to .h5
    file.

    Arguments:
    simulation_input_h5_path -- path to LISA simulation data .h5 file to
    calculate TDI channels from
    tdi_output_h5_path -- path to .h5 file to save TDI channels in
    t0 -- initial time of simulation
    dt -- time step of simulation
    """

    channels = [X2, Y2, Z2]
    tdi_names = ["X", "Y", "Z"]
    tdi_dict = TimeSeriesDict()

    for i in range(len(channels)):
        channel = channels[i]

        # GET DATA FROM LISA INSTRUMENT
        data = Data.from_instrument(tdi_output_h5_path)
        data.delay_derivative = None

        # CALCULATE TDI CHANNEL DATA
        tdi_calculator = channel.build(data.args)
        tdi_data = tdi_calculator(data.measurements)

        # WINDOW TDI CHANNEL DATA
        window = tukey(tdi_data.size, alpha=0.001)
        tdi_dict[tdi_names[i]] = TimeSeries(tdi_data * window, t0=t0, dt=dt)

    # SAVE TDI CHANNEL DATA TO FILE
    tdi_dict.write(tdi_output_h5_path, overwrite=True)


def inject_glitch(
    glitch_input_h5, glitch_input_txt, simulation_output_h5, tdi_output_h5,
    disable_noise,
):
    cl_args = init_cl()

    if cl_args.tdi_output_h5 is not None:
        glitch_input_h5 = cl_args.glitch_input_h5
        glitch_input_txt = cl_args.glitch_input_txt
        simulation_output_h5 = cl_args.simulation_output_h5
        tdi_output_h5 = cl_args.tdi_output_h5
        disable_noise = cl_args.disable_noise

    glitch_inputs = init_glitch_inputs(PATH_glitch_data + glitch_input_txt)

    simulate_lisa(
        PATH_glitch_data + glitch_input_h5,
        PATH_simulation_data + simulation_output_h5,
        glitch_inputs,
        disable_noise,
    )

    compute_and_save_tdi_channels(
        PATH_simulation_data + simulation_output_h5,
        PATH_tdi_data + tdi_output_h5,
        glitch_inputs["dt"],
        glitch_inputs["t0"],
    )


"""Uncomment to run inject_glitch alone"""
# if __name__ == "__main__":
#     inject_glitch(tdi_output_h5=None)
