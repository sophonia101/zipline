#
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc

from abc import abstractmethod
from six import with_metaclass

DEFAULT_PER_SHARE_COST = 0.0075       # 0.75 cents per share
DEFAULT_PER_CONTRACT_COST = 0.85      # $0.85 per future contract
DEFAULT_PER_DOLLAR_COST = 0.0015      # 0.15 cents per dollar
DEFAULT_MINIMUM_COST_PER_TRADE = 1.0  # $1 per trade
FUTURE_EXCHANGE_FEES_BY_SYMBOL = {    # Default fees by root symbol
    'AD': 1.60,  # AUD
    'AI': 0.96,  # Bloomberg Commodity Index
    'BD': 1.50,  # Big Dow
    'BO': 1.95,  # Soybean Oil
    'BP': 1.60,  # GBP
    'CD': 1.60,  # CAD
    'CL': 1.50,  # Crude Oil
    'CM': 1.03,  # Corn e-mini
    'CN': 1.95,  # Corn
    'DJ': 1.50,  # Dow Jones
    'EC': 1.60,  # Euro FX
    'ED': 1.25,  # Eurodollar
    'EE': 1.50,  # Euro FX e-mini
    'EI': 1.50,  # MSCI Emerging Markets mini
    'EL': 1.50,  # Eurodollar NYSE LIFFE
    'ER': 0.65,  # Russell2000 e-mini
    'ES': 1.18,  # SP500 e-mini
    'ET': 1.50,  # Ethanol
    'EU': 1.50,  # Eurodollar e-micro
    'FC': 2.03,  # Feeder Cattle
    'FF': 0.96,  # 3-Day Federal Funds
    'FI': 0.56,  # Deliverable Interest Rate Swap 5y
    'FS': 1.50,  # Interest Rate Swap 5y
    'FV': 0.65,  # US 5y
    'GC': 1.50,  # Gold
    'HG': 1.50,  # Copper
    'HO': 1.50,  # Heating Oil
    'HU': 1.50,  # Unleaded Gasoline
    'JE': 0.16,  # JPY e-mini
    'JY': 1.60,  # JPY
    'LB': 2.03,  # Lumber
    'LC': 2.03,  # Live Cattle
    'LH': 2.03,  # Lean Hogs
    'MB': 1.50,  # Municipal Bonds
    'MD': 1.50,  # SP400 Midcap
    'ME': 1.60,  # MXN
    'MG': 1.50,  # MSCI EAFE mini
    'MI': 1.18,  # SP400 Midcap e-mini
    'MS': 1.03,  # Soybean e-mini
    'MW': 1.03,  # Wheat e-mini
    'ND': 1.50,  # Nasdaq100
    'NG': 1.50,  # Natural Gas
    'NK': 2.15,  # Nikkei225
    'NQ': 1.18,  # Nasdaq100 e-mini
    'NZ': 1.60,  # NZD
    'OA': 1.95,  # Oats
    'PA': 1.50,  # Palladium
    'PB': 1.50,  # Pork Bellies
    'PL': 1.50,  # Platinum
    'QG': 0.50,  # Natural Gas e-mini
    'QM': 1.20,  # Crude Oil e-mini
    'RM': 1.50,  # Russell1000 e-mini
    'RR': 1.95,  # Rough Rice
    'SB': 2.10,  # Sugar
    'SF': 1.60,  # CHF
    'SM': 1.95,  # Soybean Meal
    'SP': 2.40,  # SP500
    'SV': 1.50,  # Silver
    'SY': 1.95,  # Soybean
    'TB': 1.50,  # Treasury Bills
    'TN': 0.56,  # Deliverable Interest Rate Swap 10y
    'TS': 1.50,  # Interest Rate Swap 10y
    'TU': 1.50,  # US 2y
    'TY': 0.75,  # US 10y
    'UB': 0.85,  # Ultra Tbond
    'US': 0.80,  # US 30y
    'VX': 1.50,  # VIX
    'WC': 1.95,  # Wheat
    'XB': 1.50,  # RBOB Gasoline
    'XG': 0.75,  # Gold e-mini
    'YM': 1.50,  # Dow Jones e-mini
    'YS': 0.75,  # Silver e-mini
}


class CommissionModel(with_metaclass(abc.ABCMeta)):
    """
    Abstract commission model interface.

    Commission models are responsible for accepting order/transaction pairs and
    calculating how much commission should be charged to an algorithm's account
    on each transaction.
    """

    @abstractmethod
    def calculate(self, order, transaction):
        """
        Calculate the amount of commission to charge on ``order`` as a result
        of ``transaction``.

        Parameters
        ----------
        order : zipline.finance.order.Order
            The order being processed.

            The ``commission`` field of ``order`` is a float indicating the
            amount of commission already charged on this order.

        transaction : zipline.finance.transaction.Transaction
            The transaction being processed. A single order may generate
            multiple transactions if there isn't enough volume in a given bar
            to fill the full amount requested in the order.

        Returns
        -------
        amount_charged : float
            The additional commission, in dollars, that we should attribute to
            this order.
        """
        raise NotImplementedError('calculate')


class EquityCommissionModel(CommissionModel):
    pass


class FutureCommissionModel(CommissionModel):
    pass


class PerUnit(object):
    """
    Mixin class for Equity and Future commission models calculated using a
    fixed cost per unit traded.
    """

    def _calculate(self, order, transaction, cost_per_unit, exchange_fee):
        """
        If there is a minimum commission:
            If the order hasn't had a commission paid yet, pay the minimum
            commission.

            If the order has paid a commission, start paying additional
            commission once the minimum commission has been reached.

        If there is no minimum commission:
            Pay commission based on number of units in the transaction.
        """
        additional_commission = abs(transaction.amount * cost_per_unit)
        min_trade_cost = self.min_trade_cost or 0

        if order.commission == 0:
            # no commission paid yet, pay at least the minimum plus a one-time
            # exchange fee.
            return max(min_trade_cost, additional_commission + exchange_fee)
        else:
            # we've already paid some commission, so figure out how much we
            # would be paying if we only counted per unit.
            per_unit_total = \
                (order.filled * cost_per_unit) + \
                additional_commission + \
                exchange_fee

            if per_unit_total < min_trade_cost:
                # if we haven't hit the minimum threshold yet, don't pay
                # additional commission
                return 0
            else:
                # we've exceeded the threshold, so pay more commission.
                return per_unit_total - order.commission


class PerShare(PerUnit, EquityCommissionModel):
    """
    Calculates a commission for a transaction based on a per share cost with
    an optional minimum cost per trade.

    Parameters
    ----------
    cost : float, optional
        The amount of commissions paid per share traded.
    min_trade_cost : float, optional
        The minimum amount of commissions paid per trade.
    """

    def __init__(self,
                 cost=DEFAULT_PER_SHARE_COST,
                 min_trade_cost=DEFAULT_MINIMUM_COST_PER_TRADE):
        self.cost_per_share = float(cost)
        self.min_trade_cost = min_trade_cost

    def __repr__(self):
        return (
            '{class_name}(cost_per_share={cost_per_share}, '
            'min_trade_cost={min_trade_cost})'
            .format(
                class_name=self.__class__.__name__,
                cost_per_share=self.cost_per_share,
                min_trade_cost=self.min_trade_cost,
            )
        )

    def calculate(self, order, transaction):
        return self._calculate(
            order, transaction, self.cost_per_share, exchange_fee=0,
        )


class PerContract(PerUnit, FutureCommissionModel):
    """
    Calculates a commission for a transaction based on a per contract cost with
    an optional minimum cost per trade.

    Parameters
    ----------
    cost_per_contract : float or dict
        The amount of commissions paid per contract traded. If given a float,
        the commission for all futures contracts is the same. If given a
        dictionary, it must map root symbols to the commission cost for
        contracts of that symbol.
    exchange_fee : float or dict
        A flat-rate fee charged by the exchange per trade. This value is a
        constant, one-time charge no matter how many contracts are being
        traded. If given a float, the fee for all contracts is the same. If
        given a dictionary, it must map root symbols to the fee for contracts
        of that symbol.
    min_trade_cost : float, optional
        The minimum amount of commissions paid per trade.
    """

    def __init__(self,
                 cost_per_contract,
                 exchange_fee,
                 min_trade_cost=DEFAULT_MINIMUM_COST_PER_TRADE):
        if isinstance(cost_per_contract, int):
            cost_per_contract = float(cost_per_contract)
        if isinstance(exchange_fee, int):
            exchange_fee = float(exchange_fee)
        self.cost_per_contract = cost_per_contract
        self.exchange_fee = exchange_fee
        self.min_trade_cost = min_trade_cost

    def __repr__(self):
        if isinstance(self.cost_per_contract, float):
            cost_per_contract = self.cost_per_contract
        else:
            cost_per_contract = '[varies]'

        if isinstance(self.exchange_fee, float):
            exchange_fee = self.exchange_fee
        else:
            exchange_fee = '[varies]'

        return (
            '{class_name}(cost_per_contract={cost_per_contract}, '
            'exchange_fee={exchange_fee}, min_trade_cost={min_trade_cost})'
            .format(
                class_name=self.__class__.__name__,
                cost_per_contract=cost_per_contract,
                exchange_fee=exchange_fee,
                min_trade_cost=self.min_trade_cost,
            )
        )

    def calculate(self, order, transaction):
        if isinstance(self.cost_per_contract, float):
            cost_per_contract = self.cost_per_contract
        else:
            # Cost per contract is a dictionary. If the user's dictionary does
            # not provide a commission cost for a certain contract, fall back
            # on the default.
            root_symbol = order.asset.root_symbol
            cost_per_contract = self.cost_per_contract.get(
                root_symbol, DEFAULT_PER_CONTRACT_COST,
            )

        if isinstance(self.exchange_fee, float):
            exchange_fee = self.exchange_fee
        else:
            # Exchange fee is a dictionary. If the user's dictionary does not
            # provide an exchange fee for a certain contract, fall back on the
            # default.
            root_symbol = order.asset.root_symbol
            exchange_fee = self.exchange_fee.get(
                root_symbol, FUTURE_EXCHANGE_FEES_BY_SYMBOL[root_symbol],
            )

        return self._calculate(
            order, transaction, cost_per_contract, exchange_fee,
        )


class PerEquityTrade(EquityCommissionModel):
    """
    Calculates a commission for a transaction based on a per trade cost.

    Parameters
    ----------
    cost : float, optional
        The flat amount of commissions paid per equity trade.
    """

    def __init__(self, cost=DEFAULT_MINIMUM_COST_PER_TRADE):
        """
        Cost parameter is the cost of a trade, regardless of share count.
        $5.00 per trade is fairly typical of discount brokers.
        """
        # Cost needs to be floating point so that calculation using division
        # logic does not floor to an integer.
        self.cost = float(cost)

    def __repr__(self):
        return '{class_name}(cost_per_trade={cost})'.format(
            class_name=self.__class__.__name__, cost=self.cost,
        )

    def calculate(self, order, transaction):
        """
        If the order hasn't had a commission paid yet, pay the fixed
        commission.
        """
        if order.commission == 0:
            # if the order hasn't had a commission attributed to it yet,
            # that's what we need to pay.
            return self.cost
        else:
            # order has already had commission attributed, so no more
            # commission.
            return 0.0


class PerFutureTrade(PerContract):
    """
    Calculates a commission for a transaction based on a per trade cost.

    Parameters
    ----------
    cost_per_trade : float or dict
        The flat amount of commissions paid per trade, regardless of the number
        of contracts being traded. If given a float, the commission for all
        futures contracts is the same. If given a dictionary, it must map root
        symbols to the commission cost for trading contracts of that symbol.
    """

    def __init__(self, cost_per_trade):
        super(PerFutureTrade, self).__init__(
            cost_per_contract=0, exchange_fee=cost_per_trade, min_trade_cost=0,
        )

    def __repr__(self):
        if isinstance(self.cost_per_trade, float):
            cost_per_trade = self.cost_per_trade
        else:
            cost_per_trade = '[varies]'
        return '{class_name}(cost_per_trade={cost_per_trade})'.format(
            class_name=self.__class__.__name__, cost_per_trade=cost_per_trade,
        )


class PerDollarBase(object):
    """
    Mixin class for Equity and Future commission models calculated using a
    fixed cost per dollar traded.
    """

    def __init__(self, cost=DEFAULT_PER_DOLLAR_COST):
        """
        Cost parameter is the cost of a trade per-dollar. 0.0015
        on $1 million means $1,500 commission (=1M * 0.0015)
        """
        self.cost_per_dollar = float(cost)

    def __repr__(self):
        return "{class_name}(cost_per_dollar={cost})".format(
            class_name=self.__class__.__name__,
            cost=self.cost_per_dollar)

    def calculate(self, order, transaction):
        """
        Pay commission based on dollar value of shares.
        """
        cost_per_share = transaction.price * self.cost_per_dollar
        return abs(transaction.amount) * cost_per_share


class PerEquityDollar(PerDollarBase, EquityCommissionModel):
    """
    Calculates a commission for a transaction based on a per dollar cost.

    Parameters
    ----------
    cost : float
        The flat amount of commissions paid per dollar of equities traded.
    """
    pass


class PerFutureDollar(PerDollarBase, FutureCommissionModel):
    """
    Calculates a commission for a transaction based on a per dollar cost.

    Parameters
    ----------
    cost : float
        The flat amount of commissions paid per dollar of futures traded.
    """
    pass


# Alias PerTrade for backwards compatibility.
PerTrade = PerEquityTrade

# Alias PerDollar for backwards compatibility.
PerDollar = PerEquityDollar
