import os
import sys
import lisaglitch
import numpy as np
import ldc.io.yml as ymlio
from ldc.utils.logging import init_logger, close_logger
from distributions import glitch_times, amplitude_dist, betas_dist

PATH_cd = os.getcwd()
PATH_lgs = os.path.abspath(os.path.join(PATH_cd, os.pardir))  # PATH to lisa_glitch_simulation directory
PATH_lisa = os.path.abspath(os.path.join(PATH_cd, os.pardir + '/' + os.pardir))
PATH_test = os.path.join(os.path.abspath(os.path.join(PATH_cd, os.pardir)), 'testing/')
PATH_io = os.path.join(PATH_lgs, 'input_output/')
PATH_tdi_out = os.path.join(PATH_lgs, 'final_tdi_outputs')


def init_cl(testing=False):
    """ initialize command line inputs """

    import argparse
    parser = argparse.ArgumentParser()
    print('-- IN init_cl --')
    # Main arguments
    parser.add_argument('--path-input', type=str, default="", help="Path to config files")
    parser.add_argument('--path-output', type=str, default="", help="Path to save output files")

    parser.add_argument('--glitch-h5-mg-output', type=str, default="glitch", help="Glitch output h5 file")
    parser.add_argument('--glitch-txt-mg-output', type=str, default="glitch", help="Glitch output txt file")
    parser.add_argument('--tdi-output-file', type=str,
                        default="final_tdi", help="Glitch output h5 file for inject_glitch")
    # parser.add_argument('--tdi-txt-output', type=str, default="", help="Glitch output txt file for inject_glitch")

    parser.add_argument('--config-input', type=str, default="ex_pipeline_cfg", help="Pipeline config file")
    parser.add_argument('--glitch-config-input', type=str, default="ex_glitch_cfg", help="Glitch config file")

    parser.add_argument('--glitch-type', type=str, default="Poisson", help="Glitch distribution")
    parser.add_argument('--glitch-amp-type', type=str, default="Gaussian", help="Glitch distribution")
    parser.add_argument('--glitch-beta-type', type=str, default="Exponential", help="Glitch distribution")
    parser.add_argument('--no-noise', type=bool, default=False, help="Whether or not to add noise")
    parser.add_argument('--no-gaps', type=bool, default=True, help="Whether or not to add gaps")

    # Testing arguments
    parser.add_argument('--testing', type=bool, default=False, help="Testing")
    parser.add_argument('--test-tmin', type=float, default=0.0, help="Testing variable: tmin")
    parser.add_argument('--test-tmax', type=float, default=6307200.0, help="Testing variable: tmax")
    parser.add_argument('--test-dt', type=float, default=5, help="Testing variable: dt")
    parser.add_argument('--test-physic-upsampling', type=float, default=1.0, help="Testing variable: physic_upsampling")
    parser.add_argument('--test-glitch-rate', type=int, default=0.2, help="Testing variable: glitch_rate")
    parser.add_argument('--test-glitch-spacing', type=int, default=20000, help="Testing variable: glitch_spacing")
    parser.add_argument('--test-avg-amp', type=float, default=10 ** -12, help="Testing variable: avg_amp")
    parser.add_argument('--test-amp-std', type=float, default=10 ** -10, help="Testing variable: amp_std")
    parser.add_argument('--test-beta-scale', type=int, default=50, help="Testing variable: amp_std")
    parser.add_argument('--test-amp-set-min', type=float, default=10 ** -10, help="Testing variable: set of amps")
    parser.add_argument('--test-amp-set-max', type=float, default=10 ** -5, help="Testing variable: set of amps")
    parser.add_argument('--test-beta-set-min', type=float, default=0.001, help="Testing variable: set of betas")
    parser.add_argument('--test-beta-set-max', type=float, default=100, help="Testing variable: set of betas")

    parser.add_argument('-l', '--log', default="", help="Log file")

    args = parser.parse_args()
    logger = init_logger(args.log, name='lisaglitch.glitch')

    if testing:
        args.testing = testing

    return args


def use_testing_inputs(args, glitch_type):
    """
    Use inputs from testing files, these are all just examples parameters for simulating lisa glitches
    """
    if glitch_type == "Poisson" or glitch_type == "poisson":
        params_dict = {'t0': args.test_tmin, 't_max':  args.test_tmax, 'dt': args.test_dt,
                       'physic_upsampling': args.test_physic_upsampling,
                       'size': args.test_tmax / args.test_dt, 'glitch_rate': args.test_glitch_rate,
                       'avg_amp': args.test_avg_amp, 'amp_std': args.test_amp_std,
                       'amp_set': [args.test_amp_set_min, args.test_amp_set_max],
                       'beta_set': [args.test_beta_set_min, args.test_beta_set_max],
                       'beta_scale': args.test_beta_scale,
                       'no_noise': args.no_noise, 'no_gaps': args.no_gaps}
    elif glitch_type == 'Equal Spacing' or glitch_type == 'equal spacing':
        params_dict = {'t0': args.test_tmin, 't_max': args.test_tmax, 'dt': args.test_dt,
                       'physic_upsampling': args.test_physic_upsampling,
                       'size': args.test_tmax / args.test_dt, 'glitch_spacing': args.test_glitch_spacing,
                       'avg_amp': args.test_avg_amp, 'amp_std': args.test_amp_std,
                       'amp_set': [args.test_amp_set_min, args.test_amp_set_max],
                       'beta_set': [args.test_beta_set_min, args.test_beta_set_max],
                       'beta_scale': args.test_beta_scale,
                       'no_noise': args.no_noise, 'no_gaps': args.no_gaps}
    else:
        sys.exit(f"Not an available distribution {glitch_type}")
    return params_dict


def simulate_glitches(glitch_type, glitch_amp_type, glitch_beta_type, params, testing=False):
    """
    Simulate glitches based off the glitch_type (the type of distribution for injected timing) and the given parameters
    params should match what's needed for the given glitch_type
    """

    # List of the injection points to sample from
    inj_points = ['tm_12', 'tm_23', 'tm_31', 'tm_13', 'tm_32', 'tm_21']

    # no_noise, no_gaps = params['no_noise'], params['no_gaps']  # TODO add this feature
    if testing:
        if glitch_amp_type == 'Set' or glitch_amp_type == 'set':
            amp_set = params['amp_set']
        else:  # Gaussian
            avg_amp, amp_std = \
                params['avg_amp'], params['amp_std']

        if glitch_beta_type == 'Set' or glitch_beta_type == 'set':
            beta_set = params['beta_set']
        else:  # Exponential
            beta_scale = params['beta_scale']
    else:
        cfg, pipe_cfg = params['cfg'], params['pipe_cfg']
        if glitch_amp_type == 'Set' or glitch_amp_type == 'set':
            amp_set = cfg['amp_set']
        else:  # Gaussian
            avg_amp, amp_std = float(cfg["avg_amp"]), float(cfg["amp_std"])

        if glitch_beta_type == 'Set' or glitch_beta_type == 'set':
            beta_set_min, beta_set_max = cfg['beta_set_min'], cfg['beta_set_max']
            beta_set = [beta_set_min, beta_set_max]
        else:  # Exponential
            beta_scale = float(cfg["beta_scale"])

    if glitch_type == "Poisson" or glitch_type == "poisson":
        # Initial inputs
        t0, t_max, dt, size, glitch_rate, output_h5, output_txt = \
            params['t0'], params['t_max'], params['dt'], params['size'], \
            params['glitch_rate'], params['glitch_h5_mg_output'], params['glitch_txt_mg_output']

        # Get times array
        timesarr = glitch_times(glitch_rate, t0, t_max, glitch_type='Poisson')
        timesarr = timesarr[timesarr < size * dt]
        number_samples = len(timesarr)

    elif glitch_type == 'Equal Spacing' or glitch_type == 'equal spacing':
        # Initial inputs
        t0, t_max, dt, size, glitch_spacing, output_h5, output_txt = \
            params['t0'], params['t_max'], params['dt'], params['size'], \
                params['glitch_spacing'], params['glitch_h5_mg_output'], params['glitch_txt_mg_output']

        timesarr = glitch_times(t0=int(t0), t_max=int(t_max), glitch_type=glitch_type, equal_space=glitch_spacing)
        timesarr = timesarr[timesarr < size * dt]
        number_samples = len(timesarr)

    else:
        sys.exit(f"Not an available distribution {glitch_type}")

    # Get Amplitudes and Betas
    if glitch_amp_type == 'Set' or glitch_amp_type == 'set':
        amp = amplitude_dist(n_samples=number_samples, type_dist='Set', amp_set=amp_set)
    else:  # Gaussian
        amp = amplitude_dist(avg_amp=float(avg_amp), std_amp=float(amp_std),
                             n_samples=number_samples, type_dist='Gaussian')

    if glitch_beta_type == 'Set' or glitch_beta_type == 'set':
        beta = betas_dist(n_samples=number_samples, beta_set=beta_set, type_dist=glitch_beta_type)
    else:  # Exponential
        beta = betas_dist(scale=float(beta_scale), n_samples=number_samples, type_dist='Exponential')

    # Produce glitches
    glitch_list = []
    for j in range(number_samples):
        print('-- Sample --', j, 'of ', number_samples)
        g = lisaglitch.IntegratedShapeletGlitch(inj_point=np.random.choice(inj_points),
                                                t0=t0, size=size, dt=dt, t_inj=timesarr[j],
                                                beta=beta[j], level=amp[j])
        glitch_list.append(g)
        g.write(path=output_h5)
        if j % 100 == 0:
            print(f"wrote {j} over {number_samples}")
        print('-- Done Sample --', j, 'of ', number_samples)

    # Make Glitch File
    print('Making Glitch File')
    header = 'generator  ' + 'size  ' + 'dt  ' + 'physics_upsampling  ' + 't0  ' + 't_inj  ' + \
             'inj_point  ' + 'beta  ' + 'level  '

    if os.path.exists(output_txt):
        os.remove(output_txt)
        print(f"The file {output_txt} has been deleted.")
    else:
        print(f"The file {output_txt} does not exist.")

    with open(f'{output_txt}', 'w') as f:
        f.write(header + "\n")
        f.close()

    with open(f'{output_txt}', 'a') as f:
        for g in glitch_list:
            f.write(str(g.generator) + "  " + str(g.size) + "  " + str(g.dt) + "  " + str(params['physic_upsampling'])
                    + "  " + str(g.t0) + "  " + str(g.t_inj) + "  " + str(g.inj_point) + "  " +
                    str(g.beta) + "  " + str(g.level) + "  " + "\n")


def main(inj_args, testing):

    print('-- INTO make_glitch main() --')

    if testing:
        args = init_cl(testing)
    else:
        args = inj_args

    if args.testing:
        glitch_type = args.glitch_type
        glitch_amp_type, glitch_beta_type = args.glitch_amp_type, args.glitch_beta_type
        params_dict = use_testing_inputs(args, glitch_type)
        params_dict['glitch_h5_mg_output'] = PATH_io + args.glitch_h5_mg_output
        params_dict['glitch_txt_mg_output'] = PATH_io + args.glitch_txt_mg_output

    else:
        cfg = ymlio.load_config(PATH_test + args.glitch_config_input)  # source config file
        pipe_cfg = ymlio.load_config(PATH_test + args.config_input)  # pipeline config file

        glitch_type = cfg["glitch_type"]
        glitch_amp_type, glitch_beta_type = cfg["amp_type"], cfg["beta_type"]

        params_dict = {'t0': cfg["t_min"].to("s").value, 't_max': cfg["t_max"].to("s").value,
                       'dt': pipe_cfg["dt_instrument"].to('s').value / pipe_cfg["physic_upsampling"],
                       'physic_upsampling': pipe_cfg["physic_upsampling"],
                       'size': cfg["t_max"].to("s").value / pipe_cfg["dt_instrument"].to('s').value,
                       'glitch_type': cfg["glitch_type"],
                       'glitch_spacing': cfg["glitch_spacing"],
                       'glitch_h5_mg_output': PATH_io + args.glitch_h5_mg_output,
                       'glitch_txt_mg_output': PATH_io + args.glitch_txt_mg_output,
                       'cfg': cfg, 'pipe_cfg': pipe_cfg}

        if glitch_type == "Poisson" or glitch_type == "poisson":
            params_dict['glitch_rate'] = cfg["glitch_rate"]
        elif glitch_type == 'Equal Spacing' or glitch_type == 'equal spacing':
            params_dict['glitch_spacing'] = cfg["glitch_spacing"]

    simulate_glitches(glitch_type, glitch_amp_type, glitch_beta_type, params_dict, testing=args.testing)

    return args.glitch_h5_mg_output, args.glitch_txt_mg_output


"""Uncomment to run make_glitch alone"""
# if __name__ == "__main__":
#
#     main(testing=False)



