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
    use_exit_signal = True   # exits are authored in custom_exit; keep the exit path enabled
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
        # Volume baselines
        dataframe['volume_avg'] = ta.SMA(dataframe, timeperiod=20, price='volume')

        # Range / volatility baselines
        dataframe['range'] = dataframe['high'] - dataframe['low']
        dataframe['range_avg'] = ta.SMA(dataframe, timeperiod=20, price='range')
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_avg'] = ta.SMA(dataframe, timeperiod=20, price='atr')

        # Momentum baseline (absolute close-to-close delta)
        dataframe['delta'] = dataframe['close'] - dataframe['close'].shift(1)
        dataframe['delta_pct'] = (dataframe['delta'] / dataframe['close'].shift(1)) * 100
        dataframe['delta_abs'] = abs(dataframe['delta'])
        dataframe['delta_avg'] = ta.SMA(dataframe, timeperiod=20, price='delta_abs')

        # Body and wick metrics for directional conviction
        dataframe['body'] = abs(dataframe['close'] - dataframe['open'])
        dataframe['upper_wick'] = dataframe['high'] - dataframe[['close', 'open']].max(axis=1)
        dataframe['lower_wick'] = dataframe[['close', 'open']].min(axis=1) - dataframe['low']

        # Compression / quiet-period detection (loosened: any 2 of 3 conditions)
        quiet_vol = dataframe['volume'] < dataframe['volume_avg'] * 1.05
        quiet_range = dataframe['range'] < dataframe['range_avg'] * 1.05
        quiet_atr = dataframe['atr'] < dataframe['atr_avg'] * 1.05
        dataframe['quiet'] = (quiet_vol.astype(int) + quiet_range.astype(int) + quiet_atr.astype(int)) >= 2
        dataframe['quiet_count'] = dataframe['quiet'].rolling(window=5).sum()

        # Expansion ratios and composite score
        dataframe['vol_ratio'] = dataframe['volume'] / dataframe['volume_avg'].replace(0, np.nan)
        dataframe['range_ratio'] = dataframe['range'] / dataframe['range_avg'].replace(0, np.nan)
        dataframe['atr_ratio'] = dataframe['atr'] / dataframe['atr_avg'].replace(0, np.nan)
        dataframe['mom_ratio'] = dataframe['delta_abs'] / dataframe['delta_avg'].replace(0, np.nan)
        dataframe['expand_score'] = (
            dataframe['vol_ratio'] + dataframe['range_ratio'] + dataframe['atr_ratio'] + dataframe['mom_ratio']
        ) / 4.0

        # RELAXED expansion event: composite score threshold + minimum move
        dataframe['min_delta_pct'] = abs(dataframe['delta_pct']) > 0.12
        dataframe['expansion_event'] = (dataframe['expand_score'] >= 1.5) & dataframe['min_delta_pct']

        # First expansion candle after a non-expansion candle
        dataframe['prev_expansion'] = dataframe['expansion_event'].shift(1).fillna(False).astype(bool)
        dataframe['first_expansion'] = dataframe['expansion_event'] & (~dataframe['prev_expansion'])

        # Ignition: first expansion with recent compression (loosened to >=1 quiet candle)
        dataframe['ignition'] = (
            dataframe['first_expansion'] &
            (dataframe['quiet_count'] >= 1)
        )

        # Cooldown: ignore if ignition fired in either of the last 2 candles (loosened from 3)
        dataframe['prev_ignition_1'] = dataframe['ignition'].shift(1).fillna(False).astype(bool)
        dataframe['prev_ignition_2'] = dataframe['ignition'].shift(2).fillna(False).astype(bool)
        dataframe['ignition'] = (
            dataframe['ignition'] &
            ~(dataframe['prev_ignition_1'] | dataframe['prev_ignition_2'])
        )

        # Readable body filter: avoid doji / indecision (relaxed vs prior wick-ratio filter)
        dataframe['readable_body'] = dataframe['body'] > (dataframe['range'] * 0.20)

        # Directional helpers
        dataframe['bull_bar'] = dataframe['close'] > dataframe['open']
        dataframe['bear_bar'] = dataframe['close'] < dataframe['open']
        dataframe['higher_close'] = dataframe['close'] > dataframe['close'].shift(1)
        dataframe['lower_close'] = dataframe['close'] < dataframe['close'].shift(1)

        # Context: ensure we aren't already in a sustained expansion (range contraction needed)
        dataframe['range_avg3'] = dataframe['range'].rolling(window=3).mean()
        dataframe['range_contracted'] = dataframe['range_avg3'] < dataframe['range_avg'] * 1.15

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # Long: bullish ignition with readable body and some compression context
        long_condition = (
            dataframe['ignition'] &
            dataframe['bull_bar'] &
            dataframe['readable_body'] &
            dataframe['higher_close'] &
            dataframe['range_contracted']
        )

        # Short: bearish ignition with readable body and some compression context
        short_condition = (
            dataframe['ignition'] &
            dataframe['bear_bar'] &
            dataframe['readable_body'] &
            dataframe['lower_close'] &
            dataframe['range_contracted']
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
        # Duration + profit are always available; the burst harvests and the hard 20-minute
        # observation cap must NOT depend on the analyzed dataframe. PRIOR BUG: the whole body
        # sat below `if dataframe.empty: return None`, so when the harness gave no dataframe
        # (every eval in practice) NOTHING fired — trades held ~110h to the stop or the shift
        # bell instead of capping at 20 minutes. The 20-min cap is the stale-thesis exit: an
        # ignition that has not paid out in 20 minutes is sold, freeing the slot.
        duration_minutes = int((current_time - trade.open_date_utc).total_seconds() // 60)

        # Scalp: harvest the initial burst quickly
        if current_profit > 0.012 and duration_minutes <= 5:
            return 'ignition_scalp'
        # Runner: expansion has evolved into a persistent move
        if current_profit > 0.025 and duration_minutes > 5:
            return 'ignition_runner'
        # Time: maximum observation window — an ignition that didn't pay is stale, recycle it
        if duration_minutes >= 20:
            return 'ignition_time'

        # Expansion-dependent reads are best-effort: if no data, treat expansion as gone so a
        # flat/underwater trade still exits early rather than waiting out the 20-minute cap.
        is_expanded = False
        if self.dp is not None:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                is_expanded = bool(dataframe.iloc[-1].get('expansion_event', False))

        # Fade: early underwater exit if expansion died immediately
        if duration_minutes <= 4 and current_profit < -0.005 and not is_expanded:
            return 'ignition_fade'
        # Early time-out: expansion died and we're flat after 8 minutes -> free the slot
        if duration_minutes >= 8 and abs(current_profit) < 0.003 and not is_expanded:
            return 'ignition_time'

        return None
