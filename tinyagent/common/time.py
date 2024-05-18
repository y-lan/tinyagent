import datetime


def get_dt_local(format="%Y-%m-%d"):
    return datetime.datetime.now().strftime(format)
