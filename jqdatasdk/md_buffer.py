# coding: utf-8
# 为jqdatasdk实现缓存功能
# 作者：王丹阳（wangdanyang@joinquant.com）
import pandas
from datetime import datetime, timedelta
from os.path import dirname, join, exists
from os import makedirs
from jqdatasdk.client import JQDataClient
from jqdatasdk.utils import (normalize_frequency,
                             convert_security, to_date_str, ParamsError)


import six
def name_convertion(s):
    security = convert_security(s)
    if isinstance(security, (str, six.string_types)):
        # 3 underscodes
        return str("___" + security.replace(".", "___"))
    elif isinstance(security, list):
        # 3 underscodes
        return [str("___" + i.replace("___", "___")) for i in security]


def fetch_data(security_code, start_date="2015-01-01", end_date="2015-12-31", frequency='daily',
                fields=None, skip_paused=False, fq='pre', count=None):
    """
    从线上与线下获取数据并完成相关操作
    """
    default_fields = ["open", "close", "low", "high", "volume", "money"]
    if fields:
        cols = ["pre_fq_factor", "post_fq_factor"].extend(fields)
    else:
        cols = ["pre_fq_factor", "post_fq_factor"].extend(default_fields)

    data = _load_data_from_cache(security_code, start_date=pandas.Timestamp(start_date).to_pydatetime() - timedelta(days=1),
                                 end_date=pandas.Timestamp(end_date).to_pydatetime() - timedelta(days=1), frequency=frequency, fields=cols,
                                 skip_paused=skip_paused, count=count)

    missing = list(sorted(set(_get_trading_time(security_code, start_date=start_date,
                                           end_date=end_date, frequency=frequency)) - set(data.index)))
    if missing:
        # 本地缓存未完全命中，获取在线数据并缓存
        # 为确保有足够的数据进行重采样，将额外获取部分数据
        origin_data = _load_online_data(security_code, start_date=(missing[0].to_pydatetime() - timedelta(days=1)),
                                        end_date=(missing[-1].to_pydatetime() + timedelta(days=1)), frequency=frequency, count=count)
        if not origin_data.empty:
            _dump_data_to_cache(security_code, origin_data, frequency)

        data = pandas.concat([data, origin_data[(origin_data.index >= missing[0]) & (origin_data.index <= missing[-1])]])
        data = data[~data.index.duplicated()]

    if skip_paused:
        data = resample_data(_process_fq(data[data["paused"] == False], fq),
                                    frequency)
    else:
        data = resample_data(_process_fq(data, fq),
                                    frequency)
        data.drop(set(data.columns) - set(fields if fields else default_fields),
                     axis=1, inplace=True)
    return data[(data.index >= start_date) & (data.index <= end_date)].round(2)


def _get_trading_time(security_code, start_date, end_date, frequency='daily'):
    """
    获取指定标的指定时段的交易时间
    :param security_code: 标的代码，仅支持单只标的
    :param start_date:
    :param end_date:
    :param frequency:
    :return:
    """
    if not isinstance(security_code, str):
        raise ValueError("仅支持单只标的代码")
    resample_mutiple, base = normalize_frequency(frequency)
    return list(JQDataClient.instance().get_price(security=security_code, start_date=start_date,
                                                    end_date=end_date, frequency={"m": "1m", "d": "1d"}[base],
                                                    fields=[], skip_paused=False, fq=None).index)


def _get_h5_archive_file(base):
    archive_file = {"d": join(dirname(__file__), "cache/jqdatasdk_cache_1d.h5"),
                    "m": join(dirname(__file__), "cache/jqdatasdk_cache_1m.h5")}[base]
    if not exists(dirname(archive_file)):
        makedirs(dirname(archive_file))
    return join(dirname(__file__), archive_file)


def _dump_data_to_cache(security_code, data, frequency):
    resample_multiple, base = normalize_frequency(frequency)
    # 去重
    with pandas.HDFStore(_get_h5_archive_file(base)) as h5_storage:
        try:
            h5_storage.remove(name_convertion(security_code),
                              where="index >= data.index[0] & index <= data.index[-1]")
        except KeyError:
            pass
        finally:
            h5_storage.append(key=name_convertion(security_code), value=data, format="table",
                data_columns=["paused"], complevel=5)


def _load_data_from_cache(security_code, start_date="2015-01-01", end_date="2015-12-31",
                          frequency='daily', fields=None, skip_paused=False, count=None):
    """
    从缓存读取数据
    :param security_code: 单只标的的字符串
    :param start_date:
    :param end_date:
    :param frequency:
    :param fields:
    :param skip_paused:
    :param count:
    :return:
    """
    scrty = convert_security(security_code)
    start = to_date_str(start_date)
    end = to_date_str(end_date) if end_date else "2015-12-31"
    if (not count) and (not start_date):
            start = "2015-01-01"
    if count and start_date:
        raise ParamsError("(start_date, count) only one param is required")

    resample_multiple, base = normalize_frequency(frequency)
    archive_file = _get_h5_archive_file(base)
    try:
        # 利用pandas自带机制过滤出需要的数据
        if not count:
            query = "index>={0} & index<={1}".format(
                pandas.Timestamp(start).to_datetime64().astype(datetime),
                pandas.Timestamp(end).to_datetime64().astype(datetime))
            cached_data = pandas.read_hdf(archive_file, key=name_convertion(scrty),
                                      where=query, columns=fields)
        else:
            query = "index<={0})".format(
                pandas.Timestamp(end).to_datetime64().astype(datetime),
                skip_paused)
            cached_data = pandas.read_hdf(archive_file, key=name_convertion(scrty),
                                      where=query, columns=fields)[-count:]
    except (KeyError, IOError):
        return pandas.DataFrame(columns=fields)
    else:
        return cached_data


def _load_online_data(security_code, start_date="2015-01-01", end_date="2015-12-31",
                      frequency='daily', count=None):
    """
    读取在线数据， 返回可供缓存的数据以及加工过的数据。
    :param security_code: 单只标的的字符串
    :param start_date:
    :param end_date:
    :param frequency:
    :param fields:
    :param skip_paused:
    :param fq:
    :param count:
    :return: (供缓存的数据， 加工过的数据)
    """
    full_fields = ["open", "close", "low", "high", "volume", "money",
                   "high_limit", "low_limit", "pre_close", "paused"]
    scrty = convert_security(security_code)
    start = to_date_str(start_date)
    end = to_date_str(end_date)
    if (not count) and (not start_date):
        start = "2015-01-01"
    if count and start_date:
        raise ParamsError("(start_date, count) only one param is required")
    resample_multiple, base = normalize_frequency(frequency)
    # 缓存的原始数据为未复权数据， 读取时本地手工复权
    origin_feq = {"m": "1m", "d": "1d"}[base]
    origin_data = JQDataClient.instance().get_price(security=scrty, start_date=start,
                                                    end_date=end, frequency=origin_feq,
                                                    fields=full_fields, skip_paused=False, fq=None, count=count)
    # 获取前复权和后复权因子， 保存至本地
    pre_fq_factor = JQDataClient.instance().get_price(security=scrty, start_date=start,
                                                    end_date=end, frequency=origin_feq,
                                                    fields=["factor"], skip_paused=False, fq="pre", count=count)
    post_fq_factor = JQDataClient.instance().get_price(security=scrty, start_date=start,
                                                    end_date=end, frequency=origin_feq,
                                                    fields=["factor"], skip_paused=False, fq="post", count=count)
    origin_data["pre_fq_factor"] = pre_fq_factor
    origin_data["post_fq_factor"] = post_fq_factor

    return origin_data


def resample_data(data, frequency):
    """
    对数据进行重采样
    :param data:
    :param frequency:
    :return:
    """
    resample_multiple, base = normalize_frequency(frequency)
    if resample_multiple == 1:
        return data
    else:
        fields_to_drop = ["factor", "high_limit", "low_limit", "pre_close", "paused"]
        data_to_resample = data.drop(set(fields_to_drop) & set(data.columns), axis=1)
        result = pandas.DataFrame()
        new_int_index = range(0, len(data_to_resample), resample_multiple)
        # 无法使用pandas自带的resample
        resamplers = {
            "open": (lambda v: [v[i] for i in new_int_index]),
            "close": (lambda v: [v[min(i + resample_multiple - 1, len(v) - 1)]  for i in new_int_index]),
            "high": (lambda v: [max(v[i:i + resample_multiple]) for i in new_int_index]),
            "low": (lambda v: [min(v[i:i + resample_multiple]) for i in new_int_index]),
            "volume": (lambda v: [sum(v[i:i + resample_multiple]) for i in new_int_index]),
            "money": (lambda v: [sum(v[i:i + resample_multiple]) for i in new_int_index]),
            "index": (lambda v: [v[min(i + resample_multiple - 1, len(v) - 1)]  for i in new_int_index])}
        for col in data_to_resample.columns:
            result[col] = resamplers[col](data_to_resample[col])
        result.index = resamplers["index"](data_to_resample.index)
        return  result


def _process_fq(data, fq):

    def _fq(df):
        if "factor" not in df:
            raise AttributeError("找不到复权因子")
        else:
            # 复权算法： value *  factor
            apply_vector = {"open": lambda v: v["open"] * v["factor"],
                            "close": lambda v: v["close"] * v["factor"],
                            "low": lambda v: v["low"] * v["factor"],
                            "high": lambda v: v["high"] * v["factor"],
                            "volume": lambda v: v["volume"] * v["factor"],
                            "money": lambda v: v["money"],
                            "factor": lambda v: v["factor"],
                            "high_limit": lambda v: v["high_limit"] * v["factor"],
                            "low_limit": lambda v: v["low_limit"] * v["factor"],
                            "pre_close": lambda v: v["pre_close"] * v["factor"],
                            "paused": lambda v: v["paused"]}
            return df.apply(lambda x: pandas.Series({label: apply_vector[label](x) for label in x.index}, index=x.index), axis=1)

    # 不复权
    if not fq:
        # 神坑：lables不可为Tuple
        result = data.drop(["pre_fq_factor", "post_fq_factor"], axis=1)
        result["factor"] = list(1 for _ in range(len(result)))
        return result
    # 前复权
    elif fq == 'pre':
        result = data.drop("post_fq_factor", axis=1)
        result.rename(columns={"pre_fq_factor": "factor"}, inplace=True)
        return _fq(result)
    # 后复权
    elif fq == "post":
        result = data.drop("pre_fq_factor", axis=1)
        result.rename(columns={"post_fq_factor": "factor"}, inplace=True)
        return _fq(result)
    else:
        raise AttributeError("不支持的复权方式")


