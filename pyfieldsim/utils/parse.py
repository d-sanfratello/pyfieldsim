from datetime import datetime


def get_filename(out_folder):
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
        f"S_{date}.h5"
    )

    return filename
