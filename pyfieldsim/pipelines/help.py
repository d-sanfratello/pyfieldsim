import argparse as ag


def main():
    parser = ag.ArgumentParser(
        prog='fs-help',
        description='This script helps navigate the order of operations to '
                    'completely simualate a stellar field image and infer '
                    'parameters over it.'
    )
    parser.add_argument("--initialize", action='store_true',
                        help="initializes the field with the true positions "
                             "and magnitudes of the stars.")
    parser.add_argument("--fs-generate-ph-noise", action='store_true',
                        help="takes a sources list and generate an exposure "
                             "with a certain time delta of observation, "
                             "taking into account the photon noise.")
    parser.add_argument("--fs-generate-background", action='store_true',
                        help="takes a field with photon noise contamination "
                             "and determines a certain level of background "
                             "noise from a SNR.")
    parser.add_argument("--fs-generate-gain-map", action='store_true',
                        help="initializes a gain map matrix for the "
                             "simulated CCD sensor.")
    parser.add_argument("--fs-generate-dark-current", action='store_true',
                        help="initializes a dark current matrix for the "
                             "simulated CCD sensor.")

    args = parser.parse_args()

    parser.print_help()


if __name__ == "__main__":
    main()
