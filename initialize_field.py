import optparse as op

from pyfieldsim.field import Field
from pyfieldsim.observation import Observation
from pyfieldsim.psf.kernels import GaussKernel


if __name__ == "__main__":
    parser = op.OptionParser()
    parser.add_option("-f", "--field-size", type='int', dest='field_size',
                      default=100,
                      help="")
    parser.add_option("-d", "--density", type='float', dest='density',
                      default=2e-3,
                      help="")
    parser.add_option("t", "--data-type", type='string', dest='datatype',
                      default='luminosity',
                      help="")
    parser.add_option("-i", "--imf-exponent", type='float', dest='e_imf',
                      default=2.4,
                      help="")
    parser.add_option("-l", "--lm-exponent", type='float', dest='e_lm',
                      default=3,
                      help="")
    parser.add_option("-c", "--lm-const", type='float', dest='cst_lm',
                      default=1,
                      help="")
    parser.add_option("-s", "--seed", type='int', dest='seed',
                      default=None,
                      help="")
    parser.add_option("-n", "--name", type='string', dest='filename',
                      default=None,
                      help="")

    (options, args) = parser.parse_args()

    if options.filename is None:
        options.filename = f"field_{options.field_size}_{options.density}"

    field = Field((options.field_size, options.field_size))
    field.initialize_field(
        density=options.density,
        e_imf=options.e_imf,
        e_lm=options.e_lm,
        cst_lm=options.cst_lm,
        seed=options.seed,
        datatype=options.datatype,
        force=False
    )

    field.export_field(filename=options.filename)


    # -------------------------------------------------------------------------
    # Complete operation
    # -------------------------------------------------------------------------
    field = Field((100, 100))
    field.initialize_field(density=0.002, datatype='luminosity')

    field.show_field('true')

    observation = Observation(field)

    # counting single stars
    stars, coords = observation.count_single_stars()
    print(f'{len(stars)} single stars')

    psf = GaussKernel(sigma=3)

    field.record_field(kernel=psf,
                       delta_time=1000, snr=10, bgnd_rel_var=0.05,
                       gain_mean=1, gain_rel_var=0.01,
                       dk_c_fraction=0.1, dk_c_rel_var=0.01, dk_c=1,
                       force=True)

    field.save_field(name='stars')
    field.show_field('exposure')
