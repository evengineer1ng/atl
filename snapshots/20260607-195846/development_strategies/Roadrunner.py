import pandas as pd
import numpy as np
from freqtrade.strategy import IStrategy

class Roadrunner(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    process_only_new_candles = True
    use_exit_signal = False
    startup_candle_count = 40

    minimal_roi = {"0": 0.05, "30": 0.025, "60": 0.01}
    stoploss = -0.03

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        prev_high = dataframe['high'].shift(1)
        prev_low = dataframe['low'].shift(1)
        prev_close = dataframe['close'].shift(1)

        tr1 = dataframe['high'] - dataframe['low']
        tr2 = abs(dataframe['high'] - prev_close)
        tr3 = abs(dataframe['low'] - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = ((dataframe['high'] - prev_high) > (prev_low - dataframe['low'])) & ((dataframe['high'] - prev_high) > 0)
        minus_dm = ((prev_low - dataframe['low']) > (dataframe['high'] - prev_high)) & ((prev_low - dataframe['low']) > 0)
        plus_dm = plus_dm.astype(float) * (dataframe['high'] - prev_high)
        minus_dm = minus_dm.astype(float) * (prev_low - dataframe['low'])

        atr = tr.rolling(window=14).mean().replace(0, np.nan)
        plus_di = 100 * plus_dm.rolling(window=14).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=14).mean() / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        dataframe['adx'] = dx.rolling(window=14).mean()

        dataframe['ema_fast'] = dataframe['close'].ewm(span=8, adjust=False).mean()
        dataframe['ema_slow'] = dataframe['close'].ewm(span=21, adjust=False).mean()
        dataframe['vol_ma'] = dataframe['volume'].rolling(window=20).mean()

        dataframe['upper_band'] = dataframe['high'].rolling(window=15).max().shift(1)
        dataframe['lower_band'] = dataframe['low'].rolling(window=15).min().shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        long_cond = (
            (dataframe['close'] > dataframe['upper_band']) &
            (dataframe['upper_band'] > dataframe['upper_band'].shift(1)) &
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['adx'] > 28) &
            (dataframe['volume'] > 1.5 * dataframe['vol_ma'])
        )

        short_cond = (
            (dataframe['close'] < dataframe['lower_band']) &
            (dataframe['lower_band'] < dataframe['lower_band'].shift(1)) &
            (dataframe['ema_fast'] < dataframe['ema_slow']) &
            (dataframe['adx'] > 28) &
            (dataframe['volume'] > 1.5 * dataframe['vol_ma'])
        )

        dataframe.loc[long_cond, 'enter_long'] = 1
        dataframe.loc[short_cond, 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
