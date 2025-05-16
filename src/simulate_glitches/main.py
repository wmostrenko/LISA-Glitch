from make_glitch import make_glitch
from inject_glitch import inject_glitch
import argparse


def init_cl():
    """Initialize commandline arguments and return Namespace object with all
    given commandline arguments.
    """

    parser = argparse.ArgumentParser()

    # FILE MANAGEMENT
    parser.add_argument(
        "--pipe_cfg_input",
        type=str,
        default="pipeline_cfg",
        help="Pipeline config file name"
    )
    parser.add_argument(
        "--glitch_config_input",
        type=str,
        default="glitch_cfg",
        help="Glitch config file name",
    )
    parser.add_argument(
        "--glitch_h5_output",
        type=str,
        default="glitch_output",
        help="Glitch output .h5 file name",
    )
    parser.add_argument(
        "--glitch_txt_output",
        type=str,
        default="glitch_output",
        help="Glitch output .txt file name",
    )
    parser.add_argument(
        "--tdi_output",
        type=str,
        default="tdi_output",
        help="TDI output .h5 file name",
    )
    parser.add_argument(
        "--simulation_output",
        type=str,
        default="simulation_output",
        help="LISA simulation output .h5 file name",
    )

    return parser.parse_args()


def main():
    args = init_cl()

    glitch_h5, glitch_txt = make_glitch(args)

    inject_glitch(glitch_h5, glitch_txt, args.tdi_output_file)


if __name__ == "__main__":
    main()
