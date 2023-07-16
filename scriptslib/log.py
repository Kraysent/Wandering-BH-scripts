from datetime import datetime


def log(*args):
    strings = []

    for arg in args:
        if isinstance(arg, datetime):
            strings.append(arg.strftime("%H:%M:%S"))
        elif isinstance(arg, float):
            strings.append(f"{arg:.03f}")
        else:
            strings.append(f"{arg}")

    print("\t".join(strings))
