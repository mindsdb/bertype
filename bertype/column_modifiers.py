""" column_modifiers.py

    Columns have "properties" (modifiers) that flat ML engines to
    treat them in a certain way. For example, a column could have
    only integer entries, but if the column is long enough compared
    to the number of unique entires, then it is said to be categorical.

    This class implements such modifiers for

        - categorical columns
        - invalid columns
        - constant columns
        - index columns
f"""
from typing import Any

import pandas


class ColumnModifier:
    """ Base class for column modifiers.
    """
    __name2id__ = {
        'nominal': 0,
        'categorical': 1,
        # 'constant': 2,
        # 'invalid': 3,
        # 'currency': 4
    }

    def __init__(self, name: str):
        """ Initializes modifier with name `name`.
        """
        self.name_ = name
        try:
            self.type_ = self.__name2id__[name]
        except KeyError as exc:
            print(f"[ERROR] column modifier {self.name_} is invalid.")
            raise KeyError("invalid column modifier.") from exc

    def get_type(self):
        """ Return modifier type as a string.
        """
        return self.type_

    def get_number_of_types(self):
        """ Return number of types.
        """
        return len(self.type_)

    @staticmethod
    def get_modifier_names() -> list:
        """ Returns list with supported types.
        """
        return list(ColumnModifier.__name2id__.keys())

    @staticmethod
    def get_number_of_modifiers() -> int:
        """ Returns number of supported types.
        """
        return len(ColumnModifier.__name2id__)

    def cast(self, x: pandas.Series, **kwargs) -> pandas.Series:
        """ Converts pandas.Series to convenient representation
            based on the modifier.
        """
        raise NotImplementedError("not implemented for base class.")


class Nominal:
    """ Columns with no special modifiers.
    """
    def __init__(self):
        """ Inializer.
        """
        super(Nominal, self).__init__('nominal')

    def cast(self, x):
        """ Returns x un-touched.
        """
        return x


class Categorical:
    """ Columns with categorical data.
    """
    def __init__(self):
        """ Inializer
        """
        super(Categorical, self).__init__('categorical')
        self.categories_ = []
        self.n_categorices = -1

    def cast(self, x: pandas.Series):
        """ Converts column type to category
        """
        x = x.astype('category')
        return x


class Invalid:
    """ Columns that engine was not able to associate with a type.
    """
    def __init__(self):
        """ Inializer.
        """
        super(Invalid, self).__init__('invalid')

    def cast(self, x):
        """ Converts all elements into NaN.
        """
        x.loc[:] = pandas.NA


class Constant:
    """ Columns where all valid entries are the same.
    """
    def __init__(self):
        """ Initializer.
        """
        super(Constant, self).__init__('constant')

    def cast(self, x, c: Any):
        """ Replaces all values in `x` by `c`.

            :param c (Any)
                Any scalar variable.
        """
        x.loc[:] = c
        return x
