from datetime import datetime


class ParseDType:
    def __call__(self, args):
        if args.datatype not in ['luminosity', 'magnitude', 'mass']:
            raise ValueError()
        elif args.datatype == 'luminosity':
            return 'L'
        elif args.datatype == 'magnitude':
            return 'm'
        elif args.datatype == 'mass':
            return 'M'


parse_dtype = ParseDType()


def get_filename(args, out_folder, dtype):
    now = datetime.utcnow()
    now = [
        getattr(now, _)
        for _ in ['year', 'month', 'day', 'hour', 'minute', 'second']
    ]
    for _ in range(6):
        if now[_] < 10:
            now[_] = f'0{now[_]}'
        else:
            now[_] = str(now[_])
    date = '-'.join(now[:-3]) + 'T' + ''.join(now[-3:])

    filename = out_folder.joinpath(
        f"{dtype}_{args.field_size}_d{args.density:.0e}_{date}"
    )

    return filename
