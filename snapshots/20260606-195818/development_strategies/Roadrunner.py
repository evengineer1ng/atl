import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy

class Roadrunner(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '5m'
    startup_candle_count = 30

    stoploss = -0.025
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    minimal_roi = {
        "0": 0.08,
        "20": 0.04,
        "40": 0.02
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_fast'] = dataframe['close'].ewm(span=9, adjust=False).mean()
        dataframe['ema_slow'] = dataframe['close'].ewm(span=21, adjust=False).mean()

        delta = dataframe['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        dataframe['rsi'] = 100 - (100 / (1 + rs))

        dataframe['dc_high'] = dataframe['high'].rolling(window=20).max().shift(1)
        dataframe['dc_low'] = dataframe['low'].rolling(window=20).min().shift(1)
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0

        long_cond = (
            (dataframe['close'] > dataframe['dc_high']) &
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['close'] > dataframe['ema_fast']) &
            (dataframe['volume'] > dataframe['volume_sma'] * 1.2) &
            (dataframe['rsi'] < 70)
        )

        short_cond = (
            (dataframe['close'] < dataframe['dc_low']) &
            (dataframe['ema_fast'] < dataframe['ema_slow']) &
            (dataframe['close'] < dataframe['ema_fast']) &
            (dataframe['volume'] > dataframe['volume_sma'] * 1.2) &
            (dataframe['rsi'] > 30)
        )

        dataframe.loc[long_cond, 'enter_long'] = 1
        dataframe.loc[short_cond, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

        exit_long = (
            (dataframe['close'] < dataframe['ema_fast']) |
            (dataframe['rsi'] > 75)
        )

        exit_short = (
            (dataframe['close'] > dataframe['ema_fast']) |
            (dataframe['rsi'] < 25)
        )

        dataframe.loc[exit_long, 'exit_long'] = 1
        dataframe.loc[exit_short, 'exit_short'] = 1

        return dataframe
