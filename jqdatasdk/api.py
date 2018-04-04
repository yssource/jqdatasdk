# coding=utf-8
from functools import wraps
from .utils import *
from .client import JQDataClient
import pandas
from datetime import timedelta
from datetime import datetime
from os.path import dirname, join, exists


@assert_auth
def get_price(security, start_date="2015-01-01", end_date="2015-12-31", frequency='daily',
    fields=None, skip_paused=False, fq='pre', count=None):
    """
    获取一支或者多只证券的行情数据

    :param security 一支证券代码或者一个证券代码的list
    :param count 与 start_date 二选一，不可同时使用.数量, 返回的结果集的行数, 即表示获取 end_date 之前几个 frequency 的数据
    :param start_date 与 count 二选一，不可同时使用. 字符串或者 datetime.datetime/datetime.date 对象, 开始时间
    :param end_date 格式同上, 结束时间, 默认是'2015-12-31', 包含此日期.
    :param frequency 单位时间长度, 几天或者几分钟, 现在支持'Xd','Xm', 'daily'(等同于'1d'), 'minute'(等同于'1m'), X是一个正整数, 分别表示X天和X分钟
    :param fields 字符串list, 默认是None(表示['open', 'close', 'high', 'low', 'volume', 'money']这几个标准字段), 支持以下属性 ['open', 'close', 'low', 'high', 'volume', 'money', 'factor', 'high_limit', 'low_limit', 'avg', 'pre_close', 'paused']
    :param skip_paused 是否跳过不交易日期(包括停牌, 未上市或者退市后的日期). 如果不跳过, 停牌时会使用停牌前的数据填充, 上市前或者退市后数据都为 nan
    :return 如果是一支证券, 则返回pandas.DataFrame对象, 行索引是datetime.datetime对象, 列索引是行情字段名字; 如果是多支证券, 则返回pandas.Panel对象, 里面是很多pandas.DataFrame对象, 索引是行情字段(open/close/…), 每个pandas.DataFrame的行索引是datetime.datetime对象, 列索引是证券代号.
    """
    scrty = convert_security(security)
    start = to_date_str(start_date)
    end = to_date_str(end_date)
    if (not count) and (not start_date):
            start = "2015-01-01"
    if count and start_date:
        raise ParamsError("(start_date, count) only one param is required")

    if start == end and fq != "post":
        # start == end一般为回测引擎调用，此时启用缓存
        if isinstance(scrty, str):
            return _fetch_data(security=scrty, start_date=start,
                               end_date=end, frequency=frequency, fields=fields,
                               skip_paused=skip_paused, fq=fq, count=count)
        else:
            data = {s: _fetch_data(security=s, start_date=start,
                               end_date=end, frequency=frequency, fields=fields,
                                   fq=fq, skip_paused=False, count=count) for s in scrty}
            return _convert_data_to_panel(data)
    else:
        # start != end一般为用户调用sdk，不启用缓存
        return JQDataClient.instance().get_price(security=scrty, start_date=start,
                                                 end_date=end, frequency=frequency, fields=fields,
                                                 skip_paused=skip_paused, fq=fq, count=count)


def _convert_data_to_panel(data):
    return pandas.Panel(data)


def _fetch_data(security, start_date="2015-01-01", end_date="2015-12-31", frequency='daily',
    fields=None, skip_paused=False, fq='pre', count=None):
    """
    从线上与线下获取数据并完成相关操作
    """
    data = _load_data_from_cache(security, start_date=start_date,
                                end_date=end_date, frequency=frequency, fields=fields,
                                skip_paused=skip_paused, fq=fq, count=count)
    if data.empty:
        # 本地缓存未命中，获取在线数据并缓存
        origin_data, data = _load_online_data(security, start_date=start_date,
                                 end_date=end_date, frequency=frequency, fields=fields,
                                 skip_paused=skip_paused, fq=fq, count=count)
        if not origin_data.empty:
            _dump_data_to_cache(security, origin_data, frequency)
    return data


def _get_h5_archive_file(base):
    archive_file = {"d": "jqdatasdk_cache_1d.h5",
                    "m": "jqdatasdk_cache_1m.h5"}[base]
    return join(dirname(__file__), archive_file)


def _dump_data_to_cache(security, data, frequency):
    resample_multiple, base = normalize_frequency(frequency)
    data.to_hdf(_get_h5_archive_file(base), key=name_convertion(security), format="table",
                data_columns=["paused"], complevel=5)


def _load_data_from_cache(security, start_date="2015-01-01", end_date="2015-12-31",
                         frequency='daily', fields=None, skip_paused=False,
                         fq='pre', count=None):
    """
    从缓存读取数据
    :param security: 单只标的的字符串
    :param start_date:
    :param end_date:
    :param frequency:
    :param fields:
    :param skip_paused:
    :param fq:
    :param count:
    :return:
    """
    scrty = convert_security(security)
    start = to_date_str(start_date)
    end = to_date_str(end_date) if end_date else "2015-12-31"
    if (not count) and (not start_date):
            start = "2015-01-01"
    if count and start_date:
        raise ParamsError("(start_date, count) only one param is required")

    resample_multiple, base = normalize_frequency(frequency)
    archive_file = _get_h5_archive_file(base)
    default_fields = ["open", "close", "low", "high", "volume", "money"]
    if fields:
        cols = ["pre_fq_factor", "post_fq_factor"] + fields
    else:
        cols = ["pre_fq_factor", "post_fq_factor"] + default_fields
    try:
        # 利用pandas自带机制过滤出需要的数据
        if not count:
            query = "index>={0} & index<={1}".format(
                pd.Timestamp(start).to_datetime64().astype(datetime),
                pd.Timestamp(end).to_datetime64().astype(datetime))
            cached_data = pd.read_hdf(archive_file, key=name_convertion(scrty),
                                      where=query, columns=cols)
        else:
            query = "index<={0})".format(
                pd.Timestamp(end).to_datetime64().astype(datetime),
                skip_paused)
            cached_data = pd.read_hdf(archive_file, key=name_convertion(scrty),
                                      where=query, columns=cols)[-count:]
    except (KeyError, IOError):
        return pandas.DataFrame(columns=cols)
    else:
        if skip_paused:
            cached_data = resample_data(_process_fq(cached_data[cached_data["paused"] == False], fq),
                                        frequency)
        else:
            cached_data = resample_data(_process_fq(cached_data, fq),
                                        frequency)
        cached_data.drop(set(cached_data.columns) - set(fields if fields else default_fields),
                         axis=1,inplace=True)
        return cached_data.round(2)


def _load_online_data(security, start_date="2015-01-01", end_date="2015-12-31",
                         frequency='daily', fields=None, skip_paused=False,
                         fq='pre', count=None):
    """
    读取在线数据， 返回可供缓存的数据以及加工过的数据。
    :param security: 单只标的的字符串
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
    scrty = convert_security(security)
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

    if skip_paused:
        data = resample_data(_process_fq(origin_data[origin_data["paused"] == False], fq),
                             frequency)

    else:
        data = resample_data(_process_fq(origin_data, fq),
                             frequency)

    if fields:
        return (origin_data, data.drop(set(data.columns) - set(fields), axis=1).round(2))
    else:
        return (origin_data, data.drop(set(data.columns) - {"open", "close", "low",
                                              "high", "volume", "money"}, axis=1).round(2))


# Todo: 处理周期重采样
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
        raise NotImplemented
        """
        fields_to_drop = ["factor", "high_limit", "low_limit", "pre_close", "paused"]
        resample_period = resample_multiple + "min" if base == "m" else base
        data_to_resample = data.drop(fields_to_drop, axis=1)
        data_to_resample["bar_time"] = list(data_to_resample.index)
        result = pandas.DataFrame()
        result["bar_time"] = data_to_resample["bar_time"].resample(resample_period).first()
        result["open"] = data_to_resample["open"].resample(resample_period).first()
        result["close"] = data_to_resample["close"].resample(resample_period).last()
        result["high"] = data_to_resample["high"].resample(resample_period).max()
        result["low"] = data_to_resample["low"].resample(resample_period).min()
        result["volume"] = data_to_resample["volume"].resample(resample_period).sum()
        result["money"] = data_to_resample["money"].resample(resample_period).sum()
        result.set_index("bar_time", inplace=True)
        result.index.name = None
        result.dropna(inplace=True)
        return  result
        """


def _process_fq(data, fq):

    def _fq(df):
        if "factor" not in df:
            raise AttributeError("找不到复权因子")
        else:
            return df.apply(lambda x: pandas.Series({"open": x["open"] * x["factor"],
                                                 "close": x["close"] * x["factor"],
                                                 "low": x["low"] * x["factor"],
                                                 "high": x["high"] * x["factor"],
                                                 "volume": x["volume"] * x["factor"],
                                                 "money": x["money"],
                                                 "factor": x["factor"],
                                                 "high_limit": x["high_limit"] * x["factor"],
                                                 "low_limit": x["low_limit"] * x["factor"],
                                                 "pre_close": x["pre_close"] * x["factor"],
                                                 "paused": x["paused"]}, index=x.index), axis=1)

    # 不复权
    if not fq:
        result = data.drop(("pre_fq_factor", "post_fq_factor"), axis=1)
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

import six
def name_convertion(s):
    security = convert_security(s)
    if isinstance(security, (str, six.string_types)):
        # 3 underscodes
        return str("___" + security.replace(".", "___"))
    elif isinstance(security, list):
        # 3 underscodes
        return [str("___" + i.replace("___", "___")) for i in security]


@assert_auth
def get_extras(info, security_list, start_date=None, end_date=None, df=True, count=None):
    """
    得到多只标的在一段时间的如下额外的数据

    :param info ['is_st', 'acc_net_value', 'unit_net_value', 'futures_sett_price', 'futures_positions'] 中的一个
    :param security_list 证券列表
    :param start_date 开始日期
    :param end_date 结束日期
    :param df 返回pandas.DataFrame对象还是一个dict
    :param count 数量, 与 start_date 二选一, 不可同时使用, 必须大于 0
    :return <df=True>:pandas.DataFrame对象, 列索引是股票代号, 行索引是datetime.datetime；<df=False>:一个dict, key是基金代号, value是numpy.ndarray
    """
    start_date = to_date_str(start_date)
    end_date = to_date_str(end_date)
    security_list = convert_security(security_list)
    return JQDataClient.instance().get_extras(**locals())


@assert_auth
def get_fundamentals(query_object, date=None, statDate=None):
    """
    查询财务数据, 详细的数据字段描述在 https://www.joinquant.com/data/dict/fundamentals 中查看

    :param query_object 一个sqlalchemy.orm.query.Query对象
    :param date 查询日期, 一个字符串(格式类似’2015-10-15’)或者datetime.date/datetime.datetime对象, 可以是None, 使用默认日期
    :param statDate: 财报统计的季度或者年份, 一个字符串, 有两种格式:1.季度: 格式是: 年 + ‘q’ + 季度序号, 例如: ‘2015q1’, ‘2013q4’. 2.年份: 格式就是年份的数字, 例如: ‘2015’, ‘2016’.
    :return 返回一个 pandas.DataFrame, 每一行对应数据库返回的每一行(可能是几个表的联合查询结果的一行), 列索引是你查询的所有字段;为了防止返回数据量过大, 我们每次最多返回10000行;当相关股票上市前、退市后，财务数据返回各字段为空
    """
    from .finance_service import get_fundamentals_sql
    if date is None and statDate is None:
        date = datetime.date.today()
        from .calendar_service import CalendarService
        trade_days = CalendarService.get_all_trade_days()
        date = list(filter(lambda item: item < date, trade_days))[-1]
    elif date:
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        date = min(to_date(date), yesterday)
    sql = get_fundamentals_sql(query_object, date, statDate)
    return JQDataClient.instance().get_fundamentals(sql=sql)


@assert_auth
def get_billboard_list(stock_list=None, start_date=None, end_date=None, count=None):
    """
    获取指定日期区间内的龙虎榜数据

    :param stock_list:一个股票代码的 list。 当值为 None 时， 返回指定日期的所有股票。
    :param start_date:开始日期
    :param end_date:结束日期
    :param count:交易日数量， 可以与 end_date 同时使用， 表示获取 end_date 前 count 个交易日的数据(含 end_date 当日)
    :return:回一个 pandas.DataFrame
    """
    start_date = to_date_str(start_date)
    end_date = to_date_str(end_date)
    stock_list = convert_security(stock_list)
    return JQDataClient.instance().get_billboard_list(**locals())


@assert_auth
def get_locked_shares(stock_list=None, start_date=None, end_date=None, forward_count=None):
    """
    获取指定日期区间内的限售解禁数据

    :param stock_list:一个股票代码的 list
    :param start_date:开始日期
    :param end_date:结束日期
    :param forward_count:交易日数量， 可以与 start_date 同时使用， 表示获取 start_date 到 forward_count 个交易日区间的数据
    :return:
    """
    start_date = to_date_str(start_date)
    end_date = to_date_str(end_date)
    stock_list = convert_security(stock_list)
    return JQDataClient.instance().get_locked_shares(**locals())


@assert_auth
def get_index_stocks(index_symbol, date=today()):
    """
    获取一个指数给定日期在平台可交易的成分股列表，请点击 https://www.joinquant.com/indexData 查看指数信息

    :param index_symbol 指数代码
    :param date: 查询日期, 一个字符串(格式类似’2015-10-15’)或者datetime.date/datetime.datetime对象, 可以是None, 使用默认日期.
    :return 股票代码的list
    """
    assert index_symbol, "index_symbol is required"
    date = to_date_str(date)
    return JQDataClient.instance().get_index_stocks(**locals())


@assert_auth
def get_industry_stocks(industry_code, date=today()):
    """
    获取在给定日期一个行业的所有股票，行业分类列表见 https://www.joinquant.com/data/dict/plateData

    :param industry_code 行业编码
    :param date 查询日期, 一个字符串(格式类似’2015-10-15’)或者datetime.date/datetime.datetime对象, 可以是None, 使用默认日期.
    :return 股票代码的list
    """
    assert industry_code, "industry_code is required"
    date = to_date_str(date)
    return JQDataClient.instance().get_industry_stocks(**locals())


@assert_auth
def get_concept_stocks(concept_code, date=today()):
    """
    获取在给定日期一个概念板块的所有股票，概念板块分类列表见 https://www.joinquant.com/data/dict/plateData

    :param concept_code 概念板块编码
    :param date: 查询日期, 一个字符串(格式类似’2015-10-15’)或者datetime.date/datetime.datetime对象, 可以是None, 使用默认日期.
    :return 股票代码的list
    """
    assert concept_code, "concept_code is required"
    date = to_date_str(date)
    return JQDataClient.instance().get_concept_stocks(**locals())


@assert_auth
def get_all_securities(types=[], date=None):
    """
    获取平台支持的所有股票、基金、指数、期货信息

    :param types list: 用来过滤securities的类型, list元素可选: ‘stock’, ‘fund’, ‘index’, ‘futures’, ‘etf’, ‘lof’, ‘fja’, ‘fjb’. types为空时返回所有股票, 不包括基金,指数和期货
    :param date 日期, 一个字符串或者 datetime.datetime/datetime.date 对象, 用于获取某日期还在上市的股票信息. 默认值为 None, 表示获取所有日期的股票信息
    :return pandas.DataFrame
    """
    date = to_date_str(date)
    return JQDataClient.instance().get_all_securities(**locals())


@assert_auth
def get_security_info(code):
    """
    获取股票/基金/指数的信息

    :param code 证券代码
    :return Security
    """
    assert code, "code is required"
    result = JQDataClient.instance().get_security_info(**locals())
    if result:
        return Security(**result)


@assert_auth
def get_all_trade_days():
    """
    获取所有交易日

    :return 包含所有交易日的 numpy.ndarray, 每个元素为一个 datetime.date 类型.
    """
    data = JQDataClient.instance().get_all_trade_days()
    return [to_date(i.item()) for i in data]


@assert_auth
def get_trade_days(start_date=None, end_date=None, count=None):
    """
    获取指定日期范围内的所有交易日

    :return numpy.ndarray, 包含指定的 start_date 和 end_date, 默认返回至 datatime.date.today() 的所有交易日
    """
    start_date = to_date_str(start_date)
    end_date = to_date_str(end_date)
    data = JQDataClient.instance().get_trade_days(**locals())
    return [to_date(i.item()) for i in data]


@assert_auth
def get_money_flow(security_list, start_date=None, end_date=None, fields=None, count=None):
    """
    获取一只或者多只股票在一个时间段内的资金流向数据

    :param security_list 一只股票代码或者一个股票代码的 list
    :param start_date 开始日期, 与 count 二选一, 不可同时使用, 一个字符串或者 datetime.datetime/datetime.date 对象, 默认为平台提供的数据的最早日期
    :param end_date 结束日期, 一个字符串或者 datetime.date/datetime.datetime 对象, 默认为 datetime.date.today()
    :param count 数量, 与 start_date 二选一，不可同时使用, 必须大于 0. 表示返回 end_date 之前 count 个交易日的数据, 包含 end_date
    :param fields 字段名或者 list, 可选
    :return pandas.DataFrame
    """
    assert security_list, "security_list is required"
    security_list = convert_security(security_list)
    start_date = to_date_str(start_date)
    end_date = to_date_str(end_date)
    return JQDataClient.instance().get_money_flow(**locals())


@assert_auth
def get_mtss(security_list, start_date=None, end_date=None, fields=None, count=None):
    """
    获取一只或者多只股票在一个时间段内的融资融券信息

    :param security_list 一只股票代码或者一个股票代码的 list
    :param start_date 开始日期, 与 count 二选一, 不可同时使用.
    :param end_date 结束日期, 一个字符串或者 datetime.date/datetime.datetime 对象, 默认为 datetime.date.today()
    :param count 数量, 与 start_date 二选一，不可同时使用, 必须大于 0. 表示返回 end_date 之前 count 个交易日的数据, 包含 end_date
    :param fields 字段名或者 list, 可选. 默认为 None, 表示取全部字段
    :return pandas.DataFrame
    """
    assert (not start_date) ^ (not count), "(start_date, count) only one param is required"
    start_date = to_date_str(start_date)
    end_date = to_date_str(end_date)
    security_list = convert_security(security_list)
    return JQDataClient.instance().get_mtss(**locals())


@assert_auth
def get_future_contracts(underlying_symbol, dt=None):
    """
    获取某期货品种在策略当前日期的可交易合约标的列表

    :param security 期货合约品种，如 ‘AG’(白银)
    :return 某期货品种在策略当前日期的可交易合约标的列表
    """
    assert underlying_symbol, "underlying_symbol is required"
    dt = to_date_str(dt)
    return JQDataClient.instance().get_future_contracts(**locals())


@assert_auth
def get_dominant_future(underlying_symbol, dt=None):
    """
    获取主力合约对应的标的

    :param security 期货合约品种，如 ‘AG’(白银)
    :return 主力合约对应的期货合约
    """
    dt = to_date_str(dt)
    return JQDataClient.instance().get_dominant_future(**locals())


@assert_auth
def get_ticks(security, end_dt=None, start_dt=None, count=None, fields=None):
    """
    获取tick数据
    :param security:
    :param end_dt:
    :param start_dt:
    :param count:
    :param fields:
    :return:
    """
    end_dt = to_date_str(end_dt)
    start_dt = to_date_str(start_dt)
    return JQDataClient.instance().get_ticks(**locals())


@assert_auth
def get_baidu_factor(category=None, day=None, stock=None, province=None):
    """
    获取百度因子搜索量数据
    :param category:数据类别，中证800的数据类别为"csi800"
    :param stock: 一只股票或一个股票list。如果为空，则包含中证800所有的成分股。
    :param day:日期，date、datetime或字符串类型。如果day为空，则返回最新的数据。
    :param province:省份名称或省份代码，如北京或110000。如果为空，则返回PC端和手机端的数据汇总。
    如果不为空，则返回指定省份的数据。
    :return:
    """
    day = to_date_str(day)
    stock = normal_security_code(stock)
    return JQDataClient.instance().get_baidu_factor(**locals())


@assert_auth
def normalize_code(code):
    """
    归一化证券代码
    :param code 如000001
    :return 证券代码的全称 如000001.XSHE
    """
    return JQDataClient.instance().normalize_code(**locals())


def read_file(path):
    """
    读取文件
    """
    with open(path, 'rb') as f:
        return f.read()


def write_file(path, content, append=False):
    """
    写入文件
    """
    if isinstance(content, six.text_type):
        content = content.encode('utf-8')
    with open(path, 'ab' if append else 'wb') as f:
        return f.write(content)


__all__ = [
            "get_price", "get_trade_days", "get_all_trade_days", "get_extras",
            "get_index_stocks", "get_industry_stocks", "get_concept_stocks", "get_all_securities",
            "get_security_info", "get_money_flow", "get_locked_shares", "get_fundamentals", "get_mtss",
            "get_future_contracts", "get_dominant_future", "normalize_code", "get_baidu_factor",
            "get_billboard_list", "get_ticks", "read_file", "write_file"]




