from fieldsim.field import Field
from fieldsim.observation import Observation
from fieldsim.psf.kernels import GaussKernel


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Complete operation
    # -------------------------------------------------------------------------
    field = Field((200, 200))
    field.initialize_field(density=0.02, datatype='luminosity')

    field.show_field('true')

    observation = Observation(field)

    # counting single stars
    stars, coords = observation.count_single_stars()

    psf = GaussKernel(sigma=3)

    field.record_field(kernel=psf,
                       delta_time=1000, snr=10, bgnd_rel_var=0.05,
                       gain_mean=1, gain_rel_var=0.01,
                       dk_c_fraction=0.1, dk_c_rel_var=0.01, dk_c=1,
                       force=True)

    field.show_field('exposure')
