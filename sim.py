from pyfieldsim.field import Field
from pyfieldsim.observation import Observation
from pyfieldsim.psf_kernels import GaussKernel


if __name__ == "__main__":
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
