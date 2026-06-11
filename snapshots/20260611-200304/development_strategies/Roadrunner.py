import pandas as pd
import numpy as np
from freqtrade.strategy import IStrategy

class Roadrunner(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    process_only_new_candles = True
    use_exit_signal = False
    startup_candle_count = 200

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

        plus_dm_raw = dataframe['high'] - prev_high
        minus_dm_raw = prev_low - dataframe['low']
        plus_dm = ((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0)) * plus_dm_raw
        minus_dm = ((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0)) * minus_dm_raw

        atr = tr.rolling(window=14).mean().replace(0, np.nan)
        plus_di = 100 * plus_dm.rolling(window=14).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=14).mean() / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)

        dataframe['atr_14'] = atr
        dataframe['plus_di'] = plus_di
        dataframe['minus_di'] = minus_di
        dataframe['adx'] = dx.rolling(window=14).mean()
        dataframe['adx_slope'] = dataframe['adx'] - dataframe['adx'].shift(2)

        dataframe['ema_8'] = dataframe['close'].ewm(span=8, adjust=False).mean()
        dataframe['ema_21'] = dataframe['close'].ewm(span=21, adjust=False).mean()
        dataframe['ema_50'] = dataframe['close'].ewm(span=50, adjust=False).mean()
        dataframe['ema_21_slope'] = dataframe['ema_21'] - dataframe['ema_21'].shift(3)

        dataframe['trend_up'] = (
            (dataframe['ema_8'] > dataframe['ema_21']) &
            (dataframe['ema_21'] > dataframe['ema_50']) &
            (dataframe['close'] > dataframe['ema_8'])
        )
        dataframe['trend_down'] = (
            (dataframe['ema_8'] < dataframe['ema_21']) &
            (dataframe['ema_21'] < dataframe['ema_50']) &
            (dataframe['close'] < dataframe['ema_8'])
        )

        dataframe['dc_high_15'] = dataframe['high'].rolling(window=15).max().shift(1)
        dataframe['dc_low_15'] = dataframe['low'].rolling(window=15).min().shift(1)

        bb_mid = dataframe['close'].rolling(window=20).mean()
        bb_std = dataframe['close'].rolling(window=20).std()
        dataframe['bb_upper'] = bb_mid + (2 * bb_std)
        dataframe['bb_lower'] = bb_mid - (2 * bb_std)
        dataframe['bb_mid'] = bb_mid
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / bb_mid.replace(0, np.nan)
        dataframe['bb_width_ma'] = dataframe['bb_width'].rolling(window=20).mean()
        dataframe['bb_squeeze'] = dataframe['bb_width'] < (dataframe['bb_width_ma'] * 0.90)
        dataframe['bb_expanding'] = dataframe['bb_width'] > dataframe['bb_width'].shift(1)
        dataframe['post_squeeze'] = (
            (dataframe['bb_squeeze'].shift(1) | dataframe['bb_squeeze'].shift(2) | dataframe['bb_squeeze'].shift(3)) &
            dataframe['bb_expanding']
        )

        dataframe['vol_ma20'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_spike'] = dataframe['volume'] > (dataframe['vol_ma20'] * 1.5)
        dataframe['volume_surge'] = dataframe['volume'] > (dataframe['volume'].shift(1) * 1.1)

        candle_range = dataframe['high'] - dataframe['low']
        candle_body = abs(dataframe['close'] - dataframe['open'])
        dataframe['strong_body'] = (candle_body / candle_range.replace(0, np.nan)) > 0.60
        dataframe['bullish'] = dataframe['close'] > dataframe['open']
        dataframe['bearish'] = dataframe['close'] < dataframe['open']

        delta = dataframe['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        dataframe['rsi'] = 100 - (100 / (1 + rs))

        dataframe['roc_3'] = (dataframe['close'] - dataframe['close'].shift(3)) / dataframe['close'].shift(3).replace(0, np.nan)

        range_5 = dataframe['high'].rolling(window=5).max() - dataframe['low'].rolling(window=5).min()
        dataframe['extended'] = range_5 > (3.0 * dataframe['atr_14'])

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        breakout_long = dataframe['close'] > (dataframe['dc_high_15'] + 0.2 * dataframe['atr_14'])
        breakout_short = dataframe['close'] < (dataframe['dc_low_15'] - 0.2 * dataframe['atr_14'])

        long_cond = (
            dataframe['trend_up'] &
            (dataframe['plus_di'] > dataframe['minus_di']) &
            (dataframe['adx'] > 30) &
            (dataframe['adx_slope'] > 0) &
            dataframe['post_squeeze'] &
            breakout_long &
            dataframe['volume_spike'] &
            dataframe['volume_surge'] &
            dataframe['bullish'] &
            dataframe['strong_body'] &
            (dataframe['rsi'] > 55) &
            (dataframe['rsi'] < 72) &
            (~dataframe['extended']) &
            (dataframe['roc_3'] > 0.005)
        )

        short_cond = (
            dataframe['trend_down'] &
            (dataframe['minus_di'] > dataframe['plus_di']) &
            (dataframe['adx'] > 30) &
            (dataframe['adx_slope'] > 0) &
            dataframe['post_squeeze'] &
            breakout_short &
            dataframe['volume_spike'] &
            dataframe['volume_surge'] &
            dataframe['bearish'] &
            dataframe['strong_body'] &
            (dataframe['rsi'] < 45) &
            (dataframe['rsi'] > 25) &
            (~dataframe['extended']) &
            (dataframe['roc_3'] < -0.005)
        )

        dataframe.loc[long_cond, 'enter_long'] = 1
        dataframe.loc[short_cond, 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
