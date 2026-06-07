from pandas import DataFrame
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
import talib.abstract as ta

class Slaking(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '4h'
    stoploss = -0.08
    use_exit_signal = True
    minimal_roi = {"0": 0.05}
    startup_candle_count = 50

    bb_period = IntParameter(20, 40, default=20, space="buy", optimize=False)
    bb_std = DecimalParameter(2.0, 3.5, default=2.5, space="buy", optimize=False)
    rsi_period = IntParameter(10, 20, default=14, space="buy", optimize=False)
    rsi_long_max = IntParameter(20, 40, default=35, space="buy", optimize=False)
    rsi_short_min = IntParameter(60, 80, default=65, space="buy", optimize=False)
    adx_period = IntParameter(10, 20, default=14, space="buy", optimize=False)
    adx_max = IntParameter(20, 35, default=25, space="buy", optimize=False)

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        middle = dataframe['close'].rolling(window=self.bb_period.value).mean()
        std = dataframe['close'].rolling(window=self.bb_period.value).std()
        dataframe['bb_middleband'] = middle
        dataframe['bb_lowerband'] = middle - self.bb_std.value * std
        dataframe['bb_upperband'] = middle + self.bb_std.value * std
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        long_cond = (
            (dataframe['close'] < dataframe['bb_lowerband']) &
            (dataframe['rsi'] < self.rsi_long_max.value) &
            (dataframe['adx'] < self.adx_max.value)
        )
        dataframe.loc[long_cond, 'enter_long'] = 1
        short_cond = (
            (dataframe['close'] > dataframe['bb_upperband']) &
            (dataframe['rsi'] > self.rsi_short_min.value) &
            (dataframe['adx'] < self.adx_max.value)
        )
        dataframe.loc[short_cond, 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        long_exit = (
            (dataframe['close'] > dataframe['bb_middleband']) |
            (dataframe['rsi'] > 55)
        )
        dataframe.loc[long_exit, 'exit_long'] = 1
        short_exit = (
            (dataframe['close'] < dataframe['bb_middleband']) |
            (dataframe['rsi'] < 45)
        )
        dataframe.loc[short_exit, 'exit_short'] = 1
        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag, side: str, **kwargs) -> float:
        return 1.0
