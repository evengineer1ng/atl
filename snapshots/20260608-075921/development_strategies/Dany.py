import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

import talib.abstract as ta
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)

class Dany(IStrategy):
    timeframe = '1m'
    can_short = True
    process_only_new_candles = True
    use_exit_signal = False
    ignore_roi_if_entry_signal = False
    exit_profit_only = False

    startup_candle_count = 50

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'emergency_exit': 'market',
    }

    stoploss = -0.03
    minimal_roi = {"0": 100.0}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Volume baseline
        dataframe['volume_avg'] = ta.SMA(dataframe['volume'], timeperiod=20)

        # Range / volatility baselines
        dataframe['range'] = dataframe['high'] - dataframe['low']
        dataframe['range_avg'] = ta.SMA(dataframe['range'], timeperiod=20)
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['atr_avg'] = ta.SMA(dataframe['atr'], timeperiod=20)

        # Momentum baseline (absolute close-to-close delta)
        dataframe['delta'] = dataframe['close'] - dataframe['close'].shift(1)
        dataframe['delta_pct'] = (dataframe['delta'] / dataframe['close'].shift(1)) * 100
        dataframe['delta_abs'] = abs(dataframe['delta'])
        dataframe['delta_avg'] = ta.SMA(dataframe['delta_abs'], timeperiod=20)

        # Body and wick metrics for directional conviction
        dataframe['body'] = abs(dataframe['close'] - dataframe['open'])
        dataframe['upper_wick'] = dataframe['high'] - dataframe[['close', 'open']].max(axis=1)
        dataframe['lower_wick'] = dataframe[['close', 'open']].min(axis=1) - dataframe['low']

        # Compression / quiet-period detection
        dataframe['quiet'] = (
            (dataframe['volume'] < dataframe['volume_avg'] * 1.0) &
            (dataframe['range'] < dataframe['range_avg'] * 1.0) &
            (dataframe['atr'] < dataframe['atr_avg'] * 1.0)
        )
        dataframe['quiet_count'] = dataframe['quiet'].rolling(window=5).sum()

        # Expansion ratios and composite score
        dataframe['vol_ratio'] = dataframe['volume'] / dataframe['volume_avg'].replace(0, np.nan)
        dataframe['range_ratio'] = dataframe['range'] / dataframe['range_avg'].replace(0, np.nan)
        dataframe['atr_ratio'] = dataframe['atr'] / dataframe['atr_avg'].replace(0, np.nan)
        dataframe['mom_ratio'] = dataframe['delta_abs'] / dataframe['delta_avg'].replace(0, np.nan)
        dataframe['expand_score'] = (
            dataframe['vol_ratio'] + dataframe['range_ratio'] + dataframe['atr_ratio'] + dataframe['mom_ratio']
        ) / 4.0

        # Expansion pillars with stricter thresholds
        dataframe['vol_expand'] = dataframe['vol_ratio'] > 1.6
        dataframe['range_expand'] = dataframe['range_ratio'] > 1.6
        dataframe['atr_expand'] = dataframe['atr_ratio'] > 1.3
        dataframe['mom_expand'] = dataframe['mom_ratio'] > 1.4
        dataframe['min_delta_pct'] = abs(dataframe['delta_pct']) > 0.15

        # Combined expansion flag
        dataframe['expansion'] = (
            dataframe['vol_expand'] &
            dataframe['range_expand'] &
            dataframe['atr_expand'] &
            dataframe['mom_expand'] &
            dataframe['min_delta_pct'] &
            (dataframe['expand_score'] >= 1.8)
        )

        # Previous expansion state
        dataframe['prev_expansion'] = dataframe['expansion'].shift(1).fillna(False).astype(bool)

        # Ignition = first expansion candle after compression (quiet period)
        dataframe['ignition'] = (
            dataframe['expansion'] &
            (~dataframe['prev_expansion']) &
            (dataframe['quiet_count'] >= 2)
        )

        # Re-ignition cooldown: ignore if any of the last 3 candles were ignition
        dataframe['prev_ignition_1'] = dataframe['ignition'].shift(1).fillna(False).astype(bool)
        dataframe['prev_ignition_2'] = dataframe['ignition'].shift(2).fillna(False).astype(bool)
        dataframe['prev_ignition_3'] = dataframe['ignition'].shift(3).fillna(False).astype(bool)
        dataframe['ignition'] = (
            dataframe['ignition'] &
            ~(dataframe['prev_ignition_1'] | dataframe['prev_ignition_2'] | dataframe['prev_ignition_3'])
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # Long: bullish ignition with body dominating upper wick and positive momentum
        long_condition = (
            dataframe['ignition'] &
            (dataframe['close'] > dataframe['open']) &
            (dataframe['close'] > dataframe['close'].shift(1)) &
            (dataframe['body'] > dataframe['upper_wick'] * 1.5) &
            (dataframe['delta_pct'] > 0)
        )

        # Short: bearish ignition with body dominating lower wick and negative momentum
        short_condition = (
            dataframe['ignition'] &
            (dataframe['close'] < dataframe['open']) &
            (dataframe['close'] < dataframe['close'].shift(1)) &
            (dataframe['body'] > dataframe['lower_wick'] * 1.5) &
            (dataframe['delta_pct'] < 0)
        )

        dataframe.loc[long_condition, 'enter_long'] = 1
        dataframe.loc[short_condition, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[str]:
        if self.dp is None:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]
        duration_minutes = int((current_time - trade.open_date_utc).total_seconds() // 60)

        is_expanded = bool(last_candle.get('expansion', False))

        # Fade: within 3 minutes, not profitable, and expansion has died
        if duration_minutes <= 3 and current_profit < -0.005 and not is_expanded:
            return 'ignition_fade'

        # Scalp: profit harvested quickly from the initial burst
        if current_profit > 0.015 and duration_minutes <= 5:
            return 'ignition_scalp'

        # Runner: expansion evolved into something more persistent
        if current_profit > 0.03 and duration_minutes > 5:
            return 'ignition_runner'

        # Time: hard observation window limit
        if duration_minutes >= 20:
            return 'ignition_time'

        return None
