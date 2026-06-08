import pandas as pd
import numpy as np
from freqtrade.strategy import IStrategy

class Roadrunner(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    process_only_new_candles = True
    use_exit_signal = False
    startup_candle_count = 100

    minimal_roi = {"0": 0.05, "30": 0.025, "60": 0.01}
    stoploss = -0.03

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

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

        dataframe['atr_14'] = atr

        dataframe['ema_slow_slope'] = dataframe['ema_slow'] - dataframe['ema_slow'].shift(3)

        dataframe['adx_slope'] = dataframe['adx'] - dataframe['adx'].shift(2)

        bb_mid = dataframe['close'].rolling(window=20).mean()
        bb_std = dataframe['close'].rolling(window=20).std()
        dataframe['bb_upper'] = bb_mid + (2 * bb_std)
        dataframe['bb_lower'] = bb_mid - (2 * bb_std)
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / bb_mid.replace(0, np.nan)
        dataframe['bb_width_expanding'] = dataframe['bb_width'] > dataframe['bb_width'].shift(3)

        dataframe['long_thrust'] = dataframe['close'] > (dataframe['upper_band'] + 0.25 * dataframe['atr_14'])
        dataframe['short_thrust'] = dataframe['close'] < (dataframe['lower_band'] - 0.25 * dataframe['atr_14'])

        inside_high = (
            (dataframe['high'].shift(1) <= dataframe['upper_band'].shift(1)).astype(int) +
            (dataframe['high'].shift(2) <= dataframe['upper_band'].shift(2)).astype(int) +
            (dataframe['high'].shift(3) <= dataframe['upper_band'].shift(3)).astype(int)
        )
        dataframe['recent_consolidation_long'] = inside_high >= 2

        inside_low = (
            (dataframe['low'].shift(1) >= dataframe['lower_band'].shift(1)).astype(int) +
            (dataframe['low'].shift(2) >= dataframe['lower_band'].shift(2)).astype(int) +
            (dataframe['low'].shift(3) >= dataframe['lower_band'].shift(3)).astype(int)
        )
        dataframe['recent_consolidation_short'] = inside_low >= 2

        dataframe['volume_spike'] = dataframe['volume'] > (1.5 * dataframe['vol_ma'])
        dataframe['volume_rising'] = dataframe['volume'] > dataframe['volume'].shift(1)

        candle_range = dataframe['high'] - dataframe['low']
        candle_body = abs(dataframe['close'] - dataframe['open'])
        dataframe['strong_candle'] = (candle_body / candle_range.replace(0, np.nan)) > 0.55

        delta = dataframe['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        dataframe['rsi'] = 100 - (100 / (1 + rs))

        dataframe['roc_3'] = (dataframe['close'] - dataframe['close'].shift(3)) / dataframe['close'].shift(3).replace(0, np.nan)

        range_5 = dataframe['high'].rolling(window=5).max() - dataframe['low'].rolling(window=5).min()
        dataframe['extended'] = range_5 > (3.5 * dataframe['atr_14'])

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        long_cond = (
            dataframe['long_thrust'] &
            (dataframe['close'] > dataframe['open']) &
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['close'] > dataframe['ema_fast']) &
            (dataframe['ema_slow_slope'] > 0) &
            (dataframe['adx'] > 28) &
            (dataframe['adx_slope'] > 0) &
            dataframe['bb_width_expanding'] &
            dataframe['volume_spike'] &
            dataframe['volume_rising'] &
            dataframe['strong_candle'] &
            (dataframe['rsi'] > 50) &
            (dataframe['rsi'] < 78) &
            dataframe['recent_consolidation_long'] &
            (~dataframe['extended']) &
            (dataframe['roc_3'] > 0.003)
        )

        short_cond = (
            dataframe['short_thrust'] &
            (dataframe['close'] < dataframe['open']) &
            (dataframe['ema_fast'] < dataframe['ema_slow']) &
            (dataframe['close'] < dataframe['ema_fast']) &
            (dataframe['ema_slow_slope'] < 0) &
            (dataframe['adx'] > 28) &
            (dataframe['adx_slope'] > 0) &
            dataframe['bb_width_expanding'] &
            dataframe['volume_spike'] &
            dataframe['volume_rising'] &
            dataframe['strong_candle'] &
            (dataframe['rsi'] < 50) &
            (dataframe['rsi'] > 22) &
            dataframe['recent_consolidation_short'] &
            (~dataframe['extended']) &
            (dataframe['roc_3'] < -0.003)
        )

        dataframe.loc[long_cond, 'enter_long'] = 1
        dataframe.loc[short_cond, 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
