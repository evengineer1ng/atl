import pandas as pd
import numpy as np
from freqtrade.strategy import IStrategy
from datetime import datetime
from typing import Optional

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

    # --- PARAMETRIC CONTROL PANEL ---
    entry_threshold = 60
    entry_threshold_boost = 50
    adx_strong = 28
    adx_med = 22
    adx_weak = 20
    rsi_cap_long = 88
    rsi_floor_short = 12
    atr_mult_breakout = 0.15
    pyramid_profit = 0.015
    pyramid_size_ratio = 0.35
    max_pyramid_entries = 2
    max_hold_bars = 60
    time_exit_profit = 0.005

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
        dataframe['ema_8_slope'] = dataframe['ema_8'] - dataframe['ema_8'].shift(2)

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
        dataframe['dc_high_10'] = dataframe['high'].rolling(window=10).max().shift(1)
        dataframe['dc_low_10'] = dataframe['low'].rolling(window=10).min().shift(1)
        dataframe['dc_high_20'] = dataframe['high'].rolling(window=20).max().shift(1)
        dataframe['dc_low_20'] = dataframe['low'].rolling(window=20).min().shift(1)

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
        dataframe['strong_body'] = (candle_body / candle_range.replace(0, np.nan)) > 0.50
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

        dataframe['seq_up'] = (dataframe['close'] > dataframe['close'].shift(1)) & (dataframe['close'].shift(1) > dataframe['close'].shift(2))
        dataframe['seq_down'] = (dataframe['close'] < dataframe['close'].shift(1)) & (dataframe['close'].shift(1) < dataframe['close'].shift(2))

        tp = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        cci_sma = tp.rolling(window=14).mean()
        mean_dev = (tp - cci_sma).abs().rolling(window=14).mean()
        dataframe['cci'] = (tp - cci_sma) / (0.015 * mean_dev.replace(0, np.nan))

        dataframe['chop_guard'] = (
            (dataframe['adx'] < 18) &
            (dataframe['bb_width'] < dataframe['bb_width_ma'] * 0.8)
        )

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # --- ENTRY GENOME ---
        # Five-axis confluence scoring with a Donchian-20 breakout bonus.
        # Dynamic threshold drops when post-squeeze volume spikes coincide,
        # intended to catch explosive moves earlier without lowering the
        # base standard across all regimes.

        # AXIS 1: TREND STRUCTURE (max 25)
        trend_full_l = dataframe['trend_up']
        trend_partial_l = (dataframe['close'] > dataframe['ema_21']) & (dataframe['ema_8'] > dataframe['ema_21'])
        trend_minimal_l = (dataframe['close'] > dataframe['ema_21']) & (dataframe['ema_21_slope'] > 0)

        score_long = trend_full_l.astype(int) * 25
        score_long += (trend_partial_l & ~trend_full_l).astype(int) * 15
        score_long += (trend_minimal_l & ~trend_partial_l & ~trend_full_l).astype(int) * 10

        trend_full_s = dataframe['trend_down']
        trend_partial_s = (dataframe['close'] < dataframe['ema_21']) & (dataframe['ema_8'] < dataframe['ema_21'])
        trend_minimal_s = (dataframe['close'] < dataframe['ema_21']) & (dataframe['ema_21_slope'] < 0)

        score_short = trend_full_s.astype(int) * 25
        score_short += (trend_partial_s & ~trend_full_s).astype(int) * 15
        score_short += (trend_minimal_s & ~trend_partial_s & ~trend_full_s).astype(int) * 10

        # AXIS 2: MOMENTUM ENGINE (max 25)
        mom_strong_l = (dataframe['adx'] > self.adx_strong) & (dataframe['plus_di'] > dataframe['minus_di']) & (dataframe['adx_slope'] > 0)
        mom_med_l = (dataframe['adx'] > self.adx_med) & (dataframe['plus_di'] > dataframe['minus_di'])
        mom_cci_l = dataframe['cci'] > 100
        score_long += mom_strong_l.astype(int) * 25
        score_long += ((mom_med_l | mom_cci_l) & ~mom_strong_l).astype(int) * 15

        mom_strong_s = (dataframe['adx'] > self.adx_strong) & (dataframe['minus_di'] > dataframe['plus_di']) & (dataframe['adx_slope'] > 0)
        mom_med_s = (dataframe['adx'] > self.adx_med) & (dataframe['minus_di'] > dataframe['plus_di'])
        mom_cci_s = dataframe['cci'] < -100
        score_short += mom_strong_s.astype(int) * 25
        score_short += ((mom_med_s | mom_cci_s) & ~mom_strong_s).astype(int) * 15

        # AXIS 3: BREAKOUT ESCALATION (max 30 with dc20 bonus)
        break_dc15_atr_l = dataframe['close'] > (dataframe['dc_high_15'] + self.atr_mult_breakout * dataframe['atr_14'])
        break_dc15_l = dataframe['close'] > dataframe['dc_high_15']
        break_dc10_l = dataframe['close'] > dataframe['dc_high_10']
        break_dc20_l = dataframe['close'] > dataframe['dc_high_20']
        seq_l = dataframe['seq_up'] & ~break_dc15_l
        score_long += break_dc15_atr_l.astype(int) * 25
        score_long += (break_dc15_l & ~break_dc15_atr_l).astype(int) * 20
        score_long += (break_dc10_l & ~break_dc15_l).astype(int) * 15
        score_long += (seq_l & ~break_dc10_l).astype(int) * 5
        score_long += break_dc20_l.astype(int) * 5

        break_dc15_atr_s = dataframe['close'] < (dataframe['dc_low_15'] - self.atr_mult_breakout * dataframe['atr_14'])
        break_dc15_s = dataframe['close'] < dataframe['dc_low_15']
        break_dc10_s = dataframe['close'] < dataframe['dc_low_10']
        break_dc20_s = dataframe['close'] < dataframe['dc_low_20']
        seq_s = dataframe['seq_down'] & ~break_dc15_s
        score_short += break_dc15_atr_s.astype(int) * 25
        score_short += (break_dc15_s & ~break_dc15_atr_s).astype(int) * 20
        score_short += (break_dc10_s & ~break_dc15_s).astype(int) * 15
        score_short += (seq_s & ~break_dc10_s).astype(int) * 5
        score_short += break_dc20_s.astype(int) * 5

        # AXIS 4: VOLUME / CANDLE CONVICTION (max 15)
        vol_strong_l = dataframe['volume_spike'] & dataframe['bullish'] & dataframe['strong_body']
        vol_med_l = (dataframe['volume'] > dataframe['volume'].shift(1)) & (dataframe['volume'] > dataframe['vol_ma20'] * 1.1) & dataframe['bullish']
        vol_base_l = dataframe['volume'] > dataframe['vol_ma20']
        score_long += vol_strong_l.astype(int) * 15
        score_long += (vol_med_l & ~vol_strong_l).astype(int) * 10
        score_long += (vol_base_l & ~vol_med_l & ~vol_strong_l).astype(int) * 5

        vol_strong_s = dataframe['volume_spike'] & dataframe['bearish'] & dataframe['strong_body']
        vol_med_s = (dataframe['volume'] > dataframe['volume'].shift(1)) & (dataframe['volume'] > dataframe['vol_ma20'] * 1.1) & dataframe['bearish']
        vol_base_s = dataframe['volume'] > dataframe['vol_ma20']
        score_short += vol_strong_s.astype(int) * 15
        score_short += (vol_med_s & ~vol_strong_s).astype(int) * 10
        score_short += (vol_base_s & ~vol_med_s & ~vol_strong_s).astype(int) * 5

        # AXIS 5: VOLATILITY REGIME (max 10)
        volreg = dataframe['post_squeeze'] | dataframe['bb_expanding']
        roc_l = dataframe['roc_3'] > 0.003
        roc_s = dataframe['roc_3'] < -0.003
        score_long += volreg.astype(int) * 10
        score_long += (roc_l & ~volreg).astype(int) * 5
        score_short += volreg.astype(int) * 10
        score_short += (roc_s & ~volreg).astype(int) * 5

        # DYNAMIC THRESHOLD: lower bar when squeeze + volume spike align
        boost_mask = dataframe['post_squeeze'] & dataframe['volume_spike']
        threshold = pd.Series(np.where(boost_mask, self.entry_threshold_boost, self.entry_threshold), index=dataframe.index)

        # DISQUALIFIERS: extremes, missing data, or chop regime
        disqual_long = (dataframe['rsi'] > self.rsi_cap_long) | dataframe['atr_14'].isna() | dataframe['chop_guard']
        disqual_short = (dataframe['rsi'] < self.rsi_floor_short) | dataframe['atr_14'].isna() | dataframe['chop_guard']

        dataframe.loc[(score_long >= threshold) & (~disqual_long), 'enter_long'] = 1
        dataframe.loc[(score_short >= threshold) & (~disqual_short), 'enter_short'] = 1

        # Expose scores and threshold for diagnostics
        dataframe['entry_score_long'] = score_long
        dataframe['entry_score_short'] = score_short
        dataframe['entry_threshold'] = threshold

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # --- EXIT GENOME (signal scaffold) ---
        # Active exit discipline is driven by custom_exit and the configured trailing/ROI engine.
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        # --- EXIT GENOME (discretionary overlay) ---
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) < 2:
            return None
        last_candle = dataframe.iloc[-1].squeeze()

        bars_since_entry = 999
        if hasattr(trade, 'open_date') and trade.open_date is not None:
            open_dt = pd.Timestamp(trade.open_date)
            bars_since_entry = int((dataframe['date'] > open_dt).sum())

        if not trade.is_short:
            if (current_profit > 0.025) and (last_candle['close'] < last_candle['ema_8']):
                return 'profit_protect_long'
            if (last_candle['close'] < last_candle['ema_21']) and (last_candle['adx'] < self.adx_weak) and (last_candle['plus_di'] < last_candle['minus_di']):
                return 'mom_death_long'
            if (current_profit < 0) and (last_candle['close'] < last_candle['ema_21']) and (last_candle['adx'] < self.adx_weak):
                return 'soft_stop_long'
            if (last_candle['rsi'] > 82) and last_candle['bearish'] and (current_profit > 0.02):
                return 'blowoff_long'
            if (bars_since_entry > self.max_hold_bars) and (current_profit < self.time_exit_profit):
                return 'time_bleed_long'
        else:
            if (current_profit > 0.025) and (last_candle['close'] > last_candle['ema_8']):
                return 'profit_protect_short'
            if (last_candle['close'] > last_candle['ema_21']) and (last_candle['adx'] < self.adx_weak) and (last_candle['minus_di'] < last_candle['plus_di']):
                return 'mom_death_short'
            if (current_profit < 0) and (last_candle['close'] > last_candle['ema_21']) and (last_candle['adx'] < self.adx_weak):
                return 'soft_stop_short'
            if (last_candle['rsi'] < 18) and last_candle['bullish'] and (current_profit > 0.02):
                return 'blowoff_short'
            if (bars_since_entry > self.max_hold_bars) and (current_profit < self.time_exit_profit):
                return 'time_bleed_short'
        return None

    def adjust_trade_position(self, trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: float, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float, **kwargs):
        # --- MANAGEMENT GENOME ---
        if current_profit <= self.pyramid_profit:
            return None

        entry_count = getattr(trade, 'nr_of_successful_entries', 1)
        if entry_count >= self.max_pyramid_entries:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe is None or len(dataframe) < 2:
            return None
        last_candle = dataframe.iloc[-1].squeeze()

        if current_entry_rate and current_rate:
            atr_pct = last_candle['atr_14'] / current_rate
            price_move = abs(current_rate - current_entry_rate) / current_entry_rate
            if price_move < (0.5 * atr_pct):
                return None

        if not trade.is_short:
            if (last_candle['trend_up'] and last_candle['adx'] > self.adx_strong and
                last_candle['plus_di'] > last_candle['minus_di'] and
                last_candle['close'] > last_candle['ema_8']):
                return trade.stake_amount * self.pyramid_size_ratio
        else:
            if (last_candle['trend_down'] and last_candle['adx'] > self.adx_strong and
                last_candle['minus_di'] > last_candle['plus_di'] and
                last_candle['close'] < last_candle['ema_8']):
                return trade.stake_amount * self.pyramid_size_ratio

        return None
