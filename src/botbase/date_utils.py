import datetime as dtm


def get_cur_dt(dt_format="%Y-%m-%d",use_utc=False):
    """
    dt_format="%Y-%m-%d"
    "%Y%m%d__%H%M%S"
    """
    cur_dt_str = dtm.datetime.utcnow()
    if not use_utc:
        cur_dt_str = cur_dt_str+dtm.timedelta(hours=8)
    cur_dt_str = cur_dt_str.strftime(dt_format)
    return cur_dt_str


def date_add_days(start_date,
                  days,
                  dt_format='%Y-%m-%d',
                  output_dt_format='%Y%m%d'):
    """
    start_date: str, date str in format dt_format
    """
    dt0 = dtm.datetime.strptime(start_date, dt_format)
    dt1 = dt0 + dtm.timedelta(days=days)
    dt1_str = dt1.strftime(output_dt_format)
    return dt1_str