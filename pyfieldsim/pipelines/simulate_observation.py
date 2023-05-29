from pyfieldsim.field import Field
from pyfieldsim.observation import Observation
from pyfieldsim.core.psf import GaussKernel

#         # Simulation advancement at each step after initialization.
#         self.add_photon_noise(
#             delta_time=delta_time,
#             force=True,
#             multiply=False
#         )
#         self.add_background(
#             fluct=background_fluct,
#             snr=snr,
#             rel_var=bgnd_rel_var,
#             force=True
#         )
#         self.apply_psf(kernel=kernel, force=True)
#
#         self.create_gain_map(
#             mean_gain=gain_mean,
#             rel_var=gain_rel_var,
#             force=True
#         )
#         self.create_dark_current(
#             b_fraction=dk_c_fraction,
#             rel_var=dk_c_rel_var,
#             dk_c=dk_c,
#             force=True
#         )
#
#         self.recorded_field = self.gain_map \
#           * self.w_psf_field \
#           + self.dark_current


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
