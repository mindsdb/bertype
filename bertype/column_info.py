"""
column_info.py

Columns may have a type, subtype and a modifier. The
Subtype matches the definition in `dtypes.py` from `type_infer`
to avoid dead kittens.
"""
from typing import Tuple

__BERTYPE__types = (
    'text',
    'number',
    'timestamp'
)

__BERTYPE__subtypes = (
    'simple_text',
    'complex_text',
    'integer',
    'float',
    'date',
    'datetime'
)
#    'text': ('simple_text', 'complex_text'),
#    'numerical': ('integer', 'float'),
#    'timestamp': ('date', 'datetime')
#}

__BERTYPE__modifiers = (
    'categorical',
    'blob',
    'constant',
    'empty',
    'identifier',
    'nominal'
)

__BERTYPE__invalid_keys = (
    'nan',
    'nat',
    'none',
    'null',
)


def get_types() -> Tuple:
    """ Returns data types.
    """
    return __BERTYPE__types


# def get_subtypes(type_name: str) -> str:
#     """ Returns data subtypes.
#     """
#     r = __BERTYPE__subtypes.get(type_name, None)
#     return r


def get_subtypes() -> Tuple:
    """ Returns data subtypes.
    """
    return __BERTYPE__subtypes


def get_modifiers() -> Tuple:
    """ Returns type modifiers.
    """
    return __BERTYPE__modifiers


def get_invalid_keys() -> Tuple:
    """ Returns types flagging invalid data.
    """
    return __BERTYPE__invalid_keys


'''
class Text(ColumnType):
    """ Implements text column type.
    """
    def __init__(self):
        """ Initializer
        """
        super(Text, self).__init__('', 'text')

    def cast(self, x):
        """ Returns __repr__ of `x`.
        """
        x = x.astype(pandas.StringDtype)
        return x

    @staticmethod
    def generate(n_elements: int,
                 min_sent: int = 1,
                 max_sent: int = 10) -> pandas.Series:
        """ Creates a text column with sentences from Lorem Ipsum.

            :param n_elements (int)
                number of elements in the Series.
            :param min_sent (int, defaults to 1)
                minimum number of sentences in an entry.
            :param max_sent (int, defaults to 10)
                maximum number of sentences in an entry.
        """
        entries = []
        for _ in range(n_elements):
            entries.append(lorem.get_sentence())
        series = pandas.Series(entries).astype(str)

        return series


class Float32(ColumnType):
    """ Implements text column type.
    """
    def __init__(self):
        """ Initializer.
        """
        super(Float32, self).__init__('text', 'float')

    @staticmethod
    def generate(n_elements, mean=0, sigma=100):
        """ Returns series of randomly distributed floating point numbers.
        """
        entries = sigma * numpy.random.randn(n_elements) + mean
        series = pandas.Series(entries).astype(numpy.float32)

        return series


class Int64(ColumnType):
    """ Implements text column type.
    """
    def __init__(self):
        """ Initializer.
        """
        super(Int64, self).__init__('text', 'float')

    @staticmethod
    def generate(n_elements, min=-1000000, max=100000):
        """ Returns series of floating point numbers.
        """
        entries = numpy.random.randint(low=min, high=max, size=n_elements)
        series = pandas.Series(entries).astype(numpy.int64)

        return series


class Timestamp(ColumnType):
    """ Implements timestamp column type.
    """
    def __init__(self):
        """ Initializer.
        """
        super(Timestamp, self).__init__('text', 'timestamp')

    @staticmethod
    def generate(n_elements):
        """ Generates a series of timestamps.
        """
        # thank you, chatGPT
        formats = [
            '%Y-%m-%d %H:%M:%S',          # 2023-12-12 15:45:30
            '%Y/%m/%d %I:%M:%S %p',       # 2023/12/12 03:45:30 PM
            '%A, %B %d, %Y %H:%M:%S',     # Monday, December 12, 2023 15:45:30
            '%a, %b %d, %Y %I:%M %p',     # Mon, Dec 12, 2023 03:45 PM
            '%d-%b-%Y %H:%M:%S',          # 12-Dec-2023 15:45:30
            '%Y%m%dT%H%M%S',              # 20231212T154530
            '%d/%m/%Y %X',                # 12/12/2023 15:45:30
            '%Y-%m-%d %I:%M %p',          # 2023-12-12 03:45 PM
            '%a, %d %b %Y %H:%M:%S %z',   # Mon, 12 Dec 2023 15:45:30 +0000
            '%Y-%m-%dT%H:%M:%S.%fZ',      # 2023-12-12T15:45:30.000000Z
            '%Y%m%d %H:%M:%S',            # 20231212 15:45:30
            '%m/%d/%Y %I:%M:%S %p',       # 12/12/2023 03:45:30 PM
            '%A, %d %B %Y %I:%M %p',      # Monday, 12 December 2023 03:45 PM
            '%a, %d %b %y %H:%M:%S',      # Mon, 12 Dec 23 15:45:30
            '%b %d, %Y %X',               # Dec 12, 2023 15:45:30
            '%Y-%m-%d %H:%M:%S.%f',       # 2023-12-12 15:45:30.000000
            '%y/%m/%d %I:%M %p',          # 23/12/12 03:45 PM
            '%A, %B %d, %y %H:%M:%S',     # Monday, December 12, 23 15:45:30
            '%a, %b %d, %y %I:%M:%S %p',  # Mon, Dec 12, 23 03:45:30 PM
            '%Y%m%d%H%M%S',               # 20231212154530
        ]
        dtfmt = str(numpy.random.choice(formats, size=1)[0])
        # hacky way of creating random dates
        start = pandas.to_datetime('1970-01-01')
        end = pandas.to_datetime('2100-01-01')
        # to nano-seconds
        start_nano = start.value // 10**9
        end_nano = end.value // 10**9
        dates_nano = numpy.random.randint(
            low=start_nano, high=end_nano, size=n_elements)
        subsec = numpy.random.randint(low=0, high=100000, size=n_elements)
        subsec_dt = [pandas.DateOffset(microsecond=x) for x in subsec]
        entries = pandas.to_datetime(dates_nano, unit='s')
        dt = [e + s for e, s in zip(entries, subsec_dt)]
        series = pandas.Series(dt).dt.strftime(dtfmt)

        return series
'''