# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'
import technical.indicators as ftt
from functools import reduce
from datetime import datetime, timezone, timedelta
from freqtrade.strategy import (
    BooleanParameter, DecimalParameter, IntParameter
)
from freqtrade.persistence import Trade
import numpy as np
from typing import Optional, Dict, List, Tuple
import threading
import requests
import time
import logging

logger = logging.getLogger(__name__)


class TheTurnstile(IStrategy):
    """
    The Turners = Cosmo/Wanda  +  a Timmy-style PRE-EXPANSION confirmation layer.

    This is an (almost) exact duplicate of the CosmoWanda strategy. ~95% of the
    behaviour is preserved unchanged:

      - Symphony LONG entry logic
      - Jorts/Wanda SHORT entry logic
      - Champion system
      - Ignition system (slot-3 delayed ignition)
      - Slot ladder / extremity gating
      - Dynamic ROI engines
      - Dynamic / champion exits
      - Ranking (power-rank) system
      - Position management (partials framework)

    The ONLY behavioural change is in entry qualification:

        Cosmo/Wanda recognises expansion AFTER it has begun.
        Timmy recognises COMPRESSION that tends to precede expansion.

        The Turners only allows an entry when BOTH agree:
          1. Cosmo/Wanda fires its expansion signal on this candle, AND
          2. A Timmy-style compression analysis of the candles immediately
             PRECEDING this candle would itself have favoured a trade in the
             SAME direction.

    Concept: "What was Timmy saying just before this move happened?"
    Expected: fewer trades, lower throughput, higher-conviction entries.

    Phase 1 is a research experiment, NOT an optimisation. We are testing:
      "Do the best Cosmo/Wanda trades tend to come from expansion events that
       Timmy would have predicted beforehand?"
    """

    ###########################################################################
    # Core
    ###########################################################################
    timeframe = '5m'
    startup_candle_count = 120
    can_short = True

    minimal_roi = {}
    stoploss = -0.499

    process_only_new_candles = False
    use_custom_exit = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    # required for partial exits
    position_adjustment_enable = True
    ###########################################################################
    # Slot-3 Delayed Ignition (per side)
    ###########################################################################
    macro_ignition_threshold = 0.06   # 6%
    micro_ignition_threshold = 0.03   # 3%

    # side -> ignition state
    _ignition: Dict[str, Dict[str, object]] = {}

    ###########################################################################
    # Extremity gating (per side): slot -> (min_green_trades, winner_profit_threshold)
    ###########################################################################
    # profit ratios (0.003 = 0.3%)
    gating_ladder: Dict[int, Tuple[int, float]] = {
        4: (1, 0.023),
        5: (2, 0.051),
        6: (3, 0.111),
        7: (4, 0.188),
        8: (5, 0.266),
        9: (6, 0.50),
    }

    ###########################################################################
    # TIMMY-STYLE PRE-EXPANSION COMPRESSION FILTER (the ONLY new system)
    ###########################################################################
    # These mirror Timmy's (LowGapBucket15m) defaults but are applied as a
    # confirmation gate on Cosmo/Wanda's existing 5m dataframe. They are plain
    # class attributes (NOT hyperopt parameters) so they do not interfere with
    # Cosmo/Wanda's optimisation spaces during this Phase-1 experiment.
    #
    # Compression regime (what Timmy enters on):
    timmy_min_gap = 0.01        # low-gap band lower bound
    timmy_max_gap = 0.07        # low-gap band upper bound
    timmy_max_adx = 15.0        # Timmy is a LOW-ADX (compression) trader
    # Conviction-score inputs:
    timmy_gain_thr = 1.0020
    timmy_loss_thr = 1.0020
    timmy_ema_fast = 3
    timmy_ema_slow = 72
    timmy_climate_window = 20
    # How many candles BEFORE the expansion candle to scan for a Timmy
    # compression signal. 6 candles == 30 minutes on the 5m timeframe.
    timmy_lookback = 6

    ###########################################################################
    # Profit siphoning (partials) framework
    ###########################################################################
    partials_enabled = IntParameter(0, 1, default=0, space="sell")
    min_partial_notional = DecimalParameter(0.0, 200.0, default=15.0, decimals=2, space="sell")

    # (profit_threshold, fraction_of_position_to_sell)
    partial_ladder: List[Tuple[float, float]] = [
        (0.01, 0.02),
        (0.02, 0.02),
        (0.03, 0.02),
        (0.05, 0.04),
        (0.08, 0.05),
    ]
    _partials_done: Dict[int, set] = {}

    ###########################################################################
    # Symphony LONG params (preserved)
    ###########################################################################
    buy_trend_above_senkou_level = DecimalParameter(1, 8, default=2, space='buy')
    buy_trend_bullish_level = DecimalParameter(1, 8, default=6, space='buy')
    buy_fan_magnitude_shift_value = DecimalParameter(1, 5, default=1, space='buy')
    buy_min_fan_magnitude_gain = DecimalParameter(1.0005, 1.01, default=1.0065, decimals=4, space='buy')

    # LONG dynamic ROI controls (preserved)
    long_min_profit = DecimalParameter(0.008, 0.02, default=0.014, space='sell')
    long_time_decay_factor = DecimalParameter(0.0001, 0.0015, default=0.001, space='sell')
    long_volume_boost_factor = DecimalParameter(0.015, 0.05, default=0.015, space='sell')
    long_adx_high_threshold = DecimalParameter(20, 35, default=23.468, space='sell')
    long_adx_low_threshold = DecimalParameter(5, 15, default=5.853, space='sell')
    long_adx_high_roi_boost = DecimalParameter(0.01, 0.05, default=0.02, space='sell')
    long_adx_low_roi_penalty = DecimalParameter(0.005, 0.02, default=0.019, space='sell')
    long_trend_confirmation_boost = DecimalParameter(0.005, 0.04, default=0.026, space='sell')
    long_trend_confirmation_penalty = DecimalParameter(0.005, 0.04, default=0.015, space='sell')

    ###########################################################################
    # Jorts/Wanda SHORT params (preserved)
    ###########################################################################
    enable_trend_filter = BooleanParameter(default=True, space='buy')
    fgi_filter_enable = BooleanParameter(default=True, space='buy')
    fgi_max_to_short = IntParameter(20, 80, default=50, space='buy')
    short_below_cloud_levels = IntParameter(1, 3, default=2, space='buy')
    di_spread_min = DecimalParameter(3.0, 25.0, default=8.0, decimals=1, space='buy')
    body_pct_min = DecimalParameter(0.4, 2.5, default=1.0, decimals=2, space='buy')
    close_below_ma_fast = BooleanParameter(default=True, space='buy')
    red_seq_len = IntParameter(1, 3, default=1, space='buy')
    vol_spike_mult = DecimalParameter(1.0, 3.0, default=1.2, decimals=2, space='buy')
    short_fan_magnitude_shift_value = DecimalParameter(1, 5, default=1, space='buy')
    short_min_fan_magnitude_loss = DecimalParameter(1.0005, 1.01, default=1.0065, decimals=4, space='buy')

    # SHORT dynamic ROI controls (preserved)
    short_min_profit = DecimalParameter(0.006, 0.03, default=0.012, space='sell')
    short_time_decay_factor = DecimalParameter(0.0002, 0.0020, default=0.0011, space='sell')
    short_volume_boost_factor = DecimalParameter(0.01, 0.08, default=0.02, space='sell')
    short_adx_high_threshold = DecimalParameter(18, 40, default=25, space='sell')
    short_adx_low_threshold = DecimalParameter(5, 15, default=8, space='sell')
    short_adx_high_roi_boost = DecimalParameter(0.005, 0.05, default=0.018, space='sell')
    short_adx_low_roi_penalty = DecimalParameter(0.003, 0.03, default=0.012, space='sell')
    short_trend_confirmation_boost = DecimalParameter(0.005, 0.05, default=0.02, space='sell')
    short_trend_confirmation_penalty = DecimalParameter(0.005, 0.05, default=0.015, space='sell')

    ###########################################################################
    # Champion / speed awareness per side
    ###########################################################################
    # ranking score = profit + velocity_weight * vel_ewma
    velocity_weight = 2.0
    velocity_alpha = 0.35

    # champion ROI boosting (rank 4+)
    champ_roi_boost_per_rank = 0.12  # multiplier step per rank band
    champ_trail_activate_base = 0.04
    champ_trail_activate_step = 0.03

    champ_allowed_drawdown_base = 0.010
    champ_allowed_drawdown_step = 0.004

    _telemetry: Dict[int, Dict[str, float]] = {}

    ###########################################################################
    # FGI cache
    ###########################################################################
    _last_fgi = 50
    _last_fgi_update = datetime.min.replace(tzinfo=timezone.utc)

    ###########################################################################
    # Optional ROI logger
    ###########################################################################
    roi_logger_enable = IntParameter(0, 1, default=0, space="sell")

    def __init__(self, config):
        super().__init__(config)
        if int(self.roi_logger_enable.value) == 1:
            threading.Thread(target=self._roi_logger, daemon=True).start()

    ###########################################################################
    # Fear & Greed Index
    ###########################################################################
    def get_fear_greed_index(self) -> int:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        if now - self._last_fgi_update < timedelta(hours=6):
            return self._last_fgi
        try:
            res = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            res.raise_for_status()
            data = res.json()
            self._last_fgi = int(data['data'][0]['value'])
            self._last_fgi_update = now
            return self._last_fgi
        except Exception:
            return self._last_fgi

    def get_scaled_fan_magnitude_gain(self) -> float:
        fgi = self.get_fear_greed_index()
        base = 1.0052
        max_gain = 1.0083
        scale = (fgi - 10) / (90 - 10)
        scale = min(max(scale, 0), 1)
        return round(base + (max_gain - base) * scale, 5)

    def get_scaled_fan_magnitude_loss(self) -> float:
        fgi = self.get_fear_greed_index()
        base = 1.0052
        max_gain = 1.0083
        scale = (fgi - 10) / (90 - 10)
        scale = min(max(scale, 0), 1)
        inverted = 1 - scale
        return round(base + (max_gain - base) * inverted, 5)

    ###########################################################################
    # Indicators (FULL MERGE + Timmy-style compression block)
    ###########################################################################
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Heikin-Ashi
        ha = qtpylib.heikinashi(df)
        df['ha_open'] = ha['open']
        df['ha_close'] = ha['close']
        df['ha_high'] = ha['high']
        df['ha_low'] = ha['low']

        # Symphony: multi-timeframe trend closes
        df['trend_close_5m'] = df['close']
        df['trend_close_15m'] = ta.EMA(df['close'], timeperiod=3)
        df['trend_close_30m'] = ta.EMA(df['close'], timeperiod=6)
        df['trend_close_1h'] = ta.EMA(df['close'], timeperiod=12)
        df['trend_close_2h'] = ta.EMA(df['close'], timeperiod=24)
        df['trend_close_4h'] = ta.EMA(df['close'], timeperiod=48)
        df['trend_close_6h'] = ta.EMA(df['close'], timeperiod=72)
        df['trend_close_8h'] = ta.EMA(df['close'], timeperiod=96)

        # Symphony: multi-timeframe trend opens
        df['trend_open_5m'] = df['open']
        df['trend_open_15m'] = ta.EMA(df['open'], timeperiod=3)
        df['trend_open_30m'] = ta.EMA(df['open'], timeperiod=6)
        df['trend_open_1h'] = ta.EMA(df['open'], timeperiod=12)
        df['trend_open_2h'] = ta.EMA(df['open'], timeperiod=24)
        df['trend_open_4h'] = ta.EMA(df['open'], timeperiod=48)
        df['trend_open_6h'] = ta.EMA(df['open'], timeperiod=72)
        df['trend_open_8h'] = ta.EMA(df['open'], timeperiod=96)

        # Fan magnitude (Symphony)
        df['fan_magnitude'] = df['trend_close_15m'] / df['trend_close_6h']
        df['fan_magnitude_gain'] = df['fan_magnitude'] / df['fan_magnitude'].shift(1)

        # Fan magnitude loss (Jorts)
        df['fan_magnitude_loss'] = df['fan_magnitude'].shift(1) / df['fan_magnitude']

        # Ichimoku cloud (Symphony)
        ichimoku = ftt.ichimoku(
            df,
            conversion_line_period=15,
            base_line_periods=45,
            laggin_span=80,
            displacement=30
        )
        df['chikou_span'] = ichimoku['chikou_span']
        df['tenkan_sen'] = ichimoku['tenkan_sen']
        df['kijun_sen'] = ichimoku['kijun_sen']
        df['senkou_a'] = ichimoku['senkou_span_a']
        df['senkou_b'] = ichimoku['senkou_span_b']
        df['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        df['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        df['cloud_green'] = ichimoku['cloud_green']
        df['cloud_red'] = ichimoku['cloud_red']

        # ATR / ADX / DI
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['plus_di'] = ta.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['minus_di'] = ta.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)

        # Volume
        df['volume_mean'] = qtpylib.rolling_mean(df['volume'], window=20)
        df['volume_change'] = df['volume'].pct_change()
        df['vol_spike'] = df['volume'] > (self.vol_spike_mult.value * df['volume_mean'])

        # Jorts EMA fast/slow
        df['ema_fast'] = ta.EMA(df['close'], timeperiod=9)
        df['ema_slow'] = ta.EMA(df['close'], timeperiod=21)
        df['close_below_fast'] = df['close'] < df['ema_fast']

        # Candle body, red streak (Jorts)
        df['is_red'] = df['ha_close'] < df['ha_open']
        df['body_pct'] = (abs(df['ha_close'] - df['ha_open']) / df['ha_open']) * 100
        red = df['is_red'].astype(int)
        df['red_streak'] = red.groupby((red != red.shift()).cumsum()).cumsum() * red

        # =====================================================================
        # TIMMY-STYLE PRE-EXPANSION COMPRESSION ANALYSIS
        # ---------------------------------------------------------------------
        # A faithful re-implementation of Timmy's (LowGapBucket15m) conviction /
        # gap / compression concepts, applied to Cosmo/Wanda's 5m dataframe.
        # All columns are prefixed `t_` so they never collide with the logic
        # above. Nothing here feeds Cosmo/Wanda's signals; it is consumed only
        # by the entry-confirmation gate in populate_entry_trend.
        # =====================================================================

        # Heikin-Ashi approximation (Timmy's own formula)
        df['t_ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2.0
        df['t_ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
        df['t_body_pct'] = (df['t_ha_close'] - df['t_ha_open']).abs() \
            / df['t_ha_open'].replace(0, np.nan) * 100.0

        # EMA fan (Timmy)
        df['t_ema_fast'] = ta.EMA(df['close'], timeperiod=int(self.timmy_ema_fast))
        df['t_ema_slow'] = ta.EMA(df['close'], timeperiod=int(self.timmy_ema_slow))
        df['t_fan'] = df['t_ema_fast'] / df['t_ema_slow']
        df['t_fan_gain'] = df['t_fan'] / df['t_fan'].shift(1)
        df['t_fan_loss'] = df['t_fan'].shift(1) / df['t_fan']

        # DI spread (reuse Cosmo's 14-period ADX/DI — identical computation)
        df['t_di_spread'] = df['minus_di'] - df['plus_di']

        # Climate temp (Timmy): rolling balance of "hot gain" vs "hot loss"
        gain_hot = (df['t_fan_gain'] > float(self.timmy_gain_thr)).rolling(
            int(self.timmy_climate_window)).mean()
        loss_hot = (df['t_fan_loss'] > float(self.timmy_loss_thr)).rolling(
            int(self.timmy_climate_window)).mean()
        df['t_temp'] = (gain_hot - loss_hot).clip(-1.0, 1.0)

        # Conviction scores (Timmy)
        temp01 = ((df['t_temp'] + 1.0) / 2.0).clip(0.0, 1.0)
        fan_gain_term = ((df['t_fan_gain'] - 1.001) / 0.004).clip(0.0, 1.0)
        adx_term = ((df['adx'] - 15.0) / 25.0).clip(0.0, 1.0)
        df['t_c_long'] = 0.45 * temp01 + 0.35 * fan_gain_term + 0.20 * adx_term

        temp01_short = ((-df['t_temp'] + 1.0) / 2.0).clip(0.0, 1.0)
        body_term = ((df['t_body_pct'] - 0.4) / 1.6).clip(0.0, 1.0)
        di_term = ((df['t_di_spread'] - 5.0) / 15.0).clip(0.0, 1.0)
        vol_term = (df['volume'] > (1.3 * df['volume_mean'])).astype(float)
        fan_loss_term = ((df['t_fan_loss'] - 1.001) / 0.004).clip(0.0, 1.0)
        df['t_c_short'] = (0.35 * temp01_short + 0.25 * body_term + 0.20 * di_term
                           + 0.10 * vol_term + 0.10 * fan_loss_term)

        # Gap + directional preference (Timmy)
        df['t_gap'] = (df['t_c_long'] - df['t_c_short']).abs()
        df['t_dir_long'] = df['t_c_long'] > df['t_c_short']

        # Compression state: low-gap band + low ADX == Timmy's entry regime.
        # This is the "pre-expansion equilibrium" Timmy looks for.
        df['t_compression'] = (
            (df['t_gap'] >= float(self.timmy_min_gap))
            & (df['t_gap'] <= float(self.timmy_max_gap))
            & (df['adx'] <= float(self.timmy_max_adx))
        )

        # Directional compression signals == what Timmy would actually have entered.
        df['t_long_signal'] = df['t_compression'] & df['t_dir_long']
        df['t_short_signal'] = df['t_compression'] & (~df['t_dir_long'])

        return df

    ###########################################################################
    # Entries (Cosmo/Wanda logic UNCHANGED + Timmy pre-expansion gate)
    ###########################################################################
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['enter_long'] = 0
        df['enter_short'] = 0

        # =========================
        # Symphony LONG entry (preserved)
        # =========================
        long_conditions = []

        # Price above Ichimoku cloud levels
        if self.buy_trend_above_senkou_level.value >= 1:
            long_conditions += [df['trend_close_5m'] > df['senkou_a'], df['trend_close_5m'] > df['senkou_b']]
        if self.buy_trend_above_senkou_level.value >= 2:
            long_conditions += [df['trend_close_15m'] > df['senkou_a'], df['trend_close_15m'] > df['senkou_b']]
        if self.buy_trend_above_senkou_level.value >= 3:
            long_conditions += [df['trend_close_30m'] > df['senkou_a'], df['trend_close_30m'] > df['senkou_b']]
        if self.buy_trend_above_senkou_level.value >= 4:
            long_conditions += [df['trend_close_1h'] > df['senkou_a'], df['trend_close_1h'] > df['senkou_b']]
        if self.buy_trend_above_senkou_level.value >= 5:
            long_conditions += [df['trend_close_2h'] > df['senkou_a'], df['trend_close_2h'] > df['senkou_b']]
        if self.buy_trend_above_senkou_level.value >= 6:
            long_conditions += [df['trend_close_4h'] > df['senkou_a'], df['trend_close_4h'] > df['senkou_b']]
        if self.buy_trend_above_senkou_level.value >= 7:
            long_conditions += [df['trend_close_6h'] > df['senkou_a'], df['trend_close_6h'] > df['senkou_b']]
        if self.buy_trend_above_senkou_level.value >= 8:
            long_conditions += [df['trend_close_8h'] > df['senkou_a'], df['trend_close_8h'] > df['senkou_b']]

        # Bullish trend stack
        if self.buy_trend_bullish_level.value >= 1:
            long_conditions.append(df['trend_close_5m'] > df['trend_open_5m'])
        if self.buy_trend_bullish_level.value >= 2:
            long_conditions.append(df['trend_close_15m'] > df['trend_open_15m'])
        if self.buy_trend_bullish_level.value >= 3:
            long_conditions.append(df['trend_close_30m'] > df['trend_open_30m'])
        if self.buy_trend_bullish_level.value >= 4:
            long_conditions.append(df['trend_close_1h'] > df['trend_open_1h'])
        if self.buy_trend_bullish_level.value >= 5:
            long_conditions.append(df['trend_close_2h'] > df['trend_open_2h'])
        if self.buy_trend_bullish_level.value >= 6:
            long_conditions.append(df['trend_close_4h'] > df['trend_open_4h'])
        if self.buy_trend_bullish_level.value >= 7:
            long_conditions.append(df['trend_close_6h'] > df['trend_open_6h'])
        if self.buy_trend_bullish_level.value >= 8:
            long_conditions.append(df['trend_close_8h'] > df['trend_open_8h'])

        # Fan magnitude (FGI-scaled)
        scaled_gain = self.get_scaled_fan_magnitude_gain()
        long_conditions.append(df['fan_magnitude_gain'] >= scaled_gain)
        long_conditions.append(df['fan_magnitude'] > 1)

        for x in range(int(self.buy_fan_magnitude_shift_value.value)):
            long_conditions.append(df['fan_magnitude'].shift(x + 1) < df['fan_magnitude'])

        if long_conditions:
            df.loc[reduce(lambda x, y: x & y, long_conditions), 'enter_long'] = 1

        # =========================
        # Jorts/Wanda SHORT entry (preserved)
        # =========================
        short_conditions = []

        fgi = self.get_fear_greed_index()
        base = 1.0052
        max_gain = 1.0083
        scale = (fgi - 10) / (90 - 10)
        scale = min(max(scale, 0), 1)
        inverted = 1 - scale
        fgi_scale = round(base + (max_gain - base) * inverted, 5)

        short_conditions.extend([
            df['is_red'],
            df['body_pct'] >= (self.body_pct_min.value * (fgi_scale / 1.0065)),
            df['vol_spike'],
            (df['minus_di'] - df['plus_di']) >= self.di_spread_min.value,
        ])

        if self.close_below_ma_fast.value:
            short_conditions.append(df['close_below_fast'])

        if self.fgi_filter_enable.value:
            short_conditions.append(fgi <= int(self.fgi_max_to_short.value))

        scaled_loss = self.get_scaled_fan_magnitude_loss()
        short_conditions.append(df['fan_magnitude_loss'] >= scaled_loss)

        for x in range(int(self.short_fan_magnitude_shift_value.value)):
            short_conditions.append(df['fan_magnitude'].shift(x + 1) > df['fan_magnitude'])

        if int(self.red_seq_len.value) > 1:
            short_conditions.append(df['red_streak'] >= int(self.red_seq_len.value))

        if short_conditions:
            df.loc[reduce(lambda x, y: x & y, short_conditions), 'enter_short'] = 1

        # =====================================================================
        # TIMMY PRE-EXPANSION CONFIRMATION GATE  (the ONLY new entry behaviour)
        # ---------------------------------------------------------------------
        # Cosmo/Wanda observes expansion (above). Timmy observed compression
        # BEFORE it. We only keep an entry when Timmy would also have favoured a
        # trade in the SAME direction in the candles immediately preceding this
        # expansion candle.
        #
        #   "What was Timmy saying before this move happened?"
        # =====================================================================
        lb = int(self.timmy_lookback)

        # Was there a directional Timmy compression signal in the `lb` candles
        # strictly BEFORE the current candle? (shift(1) excludes the current
        # expansion candle, which by definition is no longer compressed.)
        prior_long = df['t_long_signal'].astype(float).shift(1).rolling(lb, min_periods=1).max()
        prior_short = df['t_short_signal'].astype(float).shift(1).rolling(lb, min_periods=1).max()
        df['t_long_confirm'] = (prior_long >= 1).fillna(False)
        df['t_short_confirm'] = (prior_short >= 1).fillna(False)

        # Timmy directional confidence over that pre-expansion window (diagnostics):
        # the best conviction Timmy held while compressed in the matching direction.
        df['t_long_conf'] = df['t_c_long'].where(df['t_long_signal']) \
            .shift(1).rolling(lb, min_periods=1).max()
        df['t_short_conf'] = df['t_c_short'].where(df['t_short_signal']) \
            .shift(1).rolling(lb, min_periods=1).max()

        # --- Apply the gate -------------------------------------------------
        cosmo_long = df['enter_long'] == 1
        cosmo_short = df['enter_short'] == 1

        final_long = cosmo_long & df['t_long_confirm']
        final_short = cosmo_short & df['t_short_confirm']

        rejected_long = cosmo_long & ~df['t_long_confirm']
        rejected_short = cosmo_short & ~df['t_short_confirm']

        # Replace Cosmo/Wanda signals with the agreed subset.
        df['enter_long'] = final_long.astype(int)
        df['enter_short'] = final_short.astype(int)

        # =====================================================================
        # RESEARCH DIAGNOSTICS (visible in tags + logs + analyzed dataframe)
        # =====================================================================
        # Persisted diagnostic columns (inspectable via plot-dataframe / FreqUI):
        df['dx_cosmo_long'] = cosmo_long.astype(int)
        df['dx_cosmo_short'] = cosmo_short.astype(int)
        df['dx_timmy_dir'] = np.where(df['t_dir_long'], 'long', 'short')
        df['dx_reject_long'] = rejected_long.astype(int)
        df['dx_reject_short'] = rejected_short.astype(int)

        # Entry tags carry Cosmo dir, Timmy dir + Timmy confidence on accepted entries.
        if 'enter_tag' not in df.columns:
            df['enter_tag'] = ''
        df.loc[final_long, 'enter_tag'] = (
            'turners_L|cosmo=L,timmy=L,conf='
            + df.loc[final_long, 't_long_conf'].round(3).astype(str)
        )
        df.loc[final_short, 'enter_tag'] = (
            'turners_S|cosmo=S,timmy=S,conf='
            + df.loc[final_short, 't_short_conf'].round(3).astype(str)
        )

        # Summary log per pair/candle-batch: how many Cosmo signals survived /
        # were rejected by Timmy disagreement.
        n_cl, n_cs = int(cosmo_long.sum()), int(cosmo_short.sum())
        n_rl, n_rs = int(rejected_long.sum()), int(rejected_short.sum())
        if n_cl or n_cs:
            logger.info(
                "[TheTurners %s] Cosmo LONG=%d (timmy-rejected=%d, agreed=%d) | "
                "Cosmo SHORT=%d (timmy-rejected=%d, agreed=%d)",
                metadata.get('pair', '?'),
                n_cl, n_rl, n_cl - n_rl,
                n_cs, n_rs, n_cs - n_rs,
            )

        return df

    ###########################################################################
    # Exits: keep minimal (custom_exit rules)
    ###########################################################################
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['exit_long'] = 0
        df['exit_short'] = 0
        return df

    ###########################################################################
    # Extremity gating (per-side, green-count + tier winner)
    ###########################################################################
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        """
        Enforces:
          - max 9 trades per side
          - slots 1-3 always allowed
          - slots 4-9 require:
              (min green trades on same side) AND (one winner at/above tier) on same side
        """
        open_trades = Trade.get_open_trades()

        is_short_side = (side == "short")
        side_trades = [t for t in open_trades if (t.is_short == is_short_side)]

        # hard cap: 9 per side
        if len(side_trades) >= 9:
            return False

        # the slot we are trying to open now
        slot = len(side_trades) + 1

        # --- Delayed ignition bookkeeping (arms at 2 trades, re-arms on 3->2) ---
        self._maybe_arm_or_rearm_slot3_ignition(is_short_side, side_trades, current_time)

        # slots 1-2 always allowed
        if slot <= 2:
            return True

        # slot 3: special delayed ignition
        if slot == 3:
            if not self._slot3_ignition_allows(is_short_side, current_time):
                return False
            return True

        # slots 4-9 gated
        rule = self.gating_ladder.get(slot)
        if rule is None:
            return True

        min_green_needed, winner_thr = rule

        green_count = 0
        has_winner = False

        for t in side_trades:
            cp = self._current_profit_ratio(t)
            if cp is None:
                continue
            if cp > 0:
                green_count += 1
            if cp >= winner_thr:
                has_winner = True

        return (green_count >= min_green_needed) and has_winner

    ###########################################################################
    # Partial exits: framework (profit siphoning)
    ###########################################################################
    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[float]:
        if int(self.partials_enabled.value) != 1:
            return None
        if current_profit <= 0:
            return None

        # best-effort min-notional guard
        if trade.stake_amount and float(trade.stake_amount) < float(self.min_partial_notional.value):
            return None

        tid = int(trade.id)
        done = self._partials_done.setdefault(tid, set())

        for i, (pt, frac) in enumerate(self.partial_ladder):
            if i in done:
                continue
            if current_profit >= pt:
                done.add(i)
                return -float(trade.amount) * float(frac)

        return None

    ###########################################################################
    # Dynamic ROI engines (preserved)
    ###########################################################################
    def _dynamic_roi_long(self, trade: Trade, current_time: datetime) -> float:
        current_time = current_time.astimezone(timezone.utc)
        od = trade.open_date
        od = od.replace(tzinfo=timezone.utc) if od.tzinfo is None else od.astimezone(timezone.utc)

        holding_minutes = (current_time - od).total_seconds() / 60.0
        base_roi = float(self.long_min_profit.value)
        dyn = base_roi + float(self.long_time_decay_factor.value) * np.log1p(holding_minutes)

        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df is not None and not df.empty:
            c = df.iloc[-1]
            adx = float(c.get('adx', 25.0))
            plus_di = float(c.get('plus_di', 20.0))
            minus_di = float(c.get('minus_di', 20.0))
            volume_change = float(c.get('volume_change', 0.0))

            dyn += float(self.long_adx_high_roi_boost.value) * (adx / 100.0)
            dyn -= float(self.long_adx_low_roi_penalty.value) * (1.0 - (adx / 100.0))

            di_strength = (plus_di - minus_di) / 100.0
            dyn += float(self.long_trend_confirmation_boost.value) * di_strength
            dyn -= float(self.long_trend_confirmation_penalty.value) * (-di_strength)

            dyn += float(self.long_volume_boost_factor.value) * np.tanh(volume_change)

        return max(dyn, 0.01)

    def _dynamic_roi_short(self, trade: Trade, current_time: datetime) -> float:
        current_time = current_time.astimezone(timezone.utc)
        od = trade.open_date
        od = od.replace(tzinfo=timezone.utc) if od.tzinfo is None else od.astimezone(timezone.utc)

        holding_minutes = (current_time - od).total_seconds() / 60.0
        base = float(self.short_min_profit.value)
        dyn = base + float(self.short_time_decay_factor.value) * np.log1p(holding_minutes)

        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df is not None and not df.empty:
            c = df.iloc[-1]
            adx = float(c.get('adx', 25.0))
            plus_di = float(c.get('plus_di', 18.0))
            minus_di = float(c.get('minus_di', 22.0))
            volchg = (float(c.get('volume', 1.0)) / max(float(c.get('volume_mean', 1.0)), 1.0)) - 1.0

            dyn += float(self.short_adx_high_roi_boost.value) * (adx / 100.0)
            dyn -= float(self.short_adx_low_roi_penalty.value) * (1.0 - (adx / 100.0))

            di_strength = (minus_di - plus_di) / 100.0
            dyn += float(self.short_trend_confirmation_boost.value) * max(di_strength, 0.0)
            dyn -= float(self.short_trend_confirmation_penalty.value) * max(-di_strength, 0.0)

            dyn += float(self.short_volume_boost_factor.value) * np.tanh(volchg)

        return max(dyn, 0.005)

    ###########################################################################
    # Telemetry + ranking (per side)
    ###########################################################################
    def _update_telemetry(self, trade: Trade, current_time: datetime, current_profit: float) -> Dict[str, float]:
        tid = int(trade.id)
        tel = self._telemetry.get(tid, {"last": current_profit, "ts": None, "vel": 0.0, "max": current_profit})

        ts = current_time.timestamp()
        if tel["ts"] is not None:
            dp = float(current_profit) - float(tel["last"])
            dt = float(ts) - float(tel["ts"])
            inst_vel = (dp / dt) if dt > 0 else 0.0
            tel["vel"] = (self.velocity_alpha * inst_vel) + ((1.0 - self.velocity_alpha) * float(tel["vel"]))

        tel["last"] = float(current_profit)
        tel["ts"] = float(ts)
        tel["max"] = max(float(tel["max"]), float(current_profit))

        self._telemetry[tid] = tel
        return tel

    def _power_ranks_side(self, side_trades: List[Trade]) -> Dict[int, int]:
        scored: List[Tuple[int, float]] = []

        for t in side_trades:
            cp = self._current_profit_ratio(t)
            if cp is None:
                continue
            tel = self._telemetry.get(int(t.id), {"vel": 0.0})
            score = float(cp) + (self.velocity_weight * float(tel.get("vel", 0.0)))
            scored.append((int(t.id), score))

        scored.sort(key=lambda x: x[1])  # worst -> best
        return {tid: i + 1 for i, (tid, _) in enumerate(scored)}

    ###########################################################################
    # Custom exit: bottom 3 scalpers, ranks 4-9 champions (per side)
    ###########################################################################
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[str]:
        # Never sell red
        if current_profit <= 0:
            self._update_telemetry(trade, current_time, current_profit)
            return None

        open_trades = Trade.get_open_trades()
        side_trades = [t for t in open_trades if (t.is_short == trade.is_short)]

        ranks = self._power_ranks_side(side_trades)
        rank = ranks.get(int(trade.id), 1)

        tel = self._update_telemetry(trade, current_time, current_profit)

        # Side-specific dynamic ROI
        dyn_roi = self._dynamic_roi_short(trade, current_time) if trade.is_short else self._dynamic_roi_long(trade, current_time)

        # Rank 1-3: pure scalper dynamic ROI
        if rank <= 3:
            if current_profit >= dyn_roi:
                return "dynamic_roi_hit"
            return None

        # Rank 4-9: champions
        champ_index = min(rank - 4, 5)  # 4..9 -> 0..5
        # Boost ROI expectation for champions
        _, rung_target = self.gating_ladder.get(rank, (None, None))

        if rung_target is not None:
            boosted_roi = max(
                dyn_roi,
                dyn_roi + 0.6 * (rung_target - dyn_roi)
            )
        else:
            boosted_roi = dyn_roi
        # Champion dynamic ROI exit (still profit-only)
        if current_profit >= boosted_roi:
            return "champ_dynamic_roi_hit"

        # Champion trailing stop-profit (let winners breathe but lock gains)
        activate_at = self.champ_trail_activate_base + (champ_index * self.champ_trail_activate_step)

        if current_profit >= activate_at:
            allowed_dd = self.champ_allowed_drawdown_base + (champ_index * self.champ_allowed_drawdown_step)

            # If profit velocity turns negative -> tighten drawdown aggressively
            if float(tel.get("vel", 0.0)) < 0.0:
                tighten = min(0.75, abs(float(tel["vel"])) * self.velocity_weight)
                allowed_dd *= (1.0 - tighten)

            # Exit if falling off peak beyond allowed drawdown (still > 0 always)
            if (float(tel["max"]) - float(current_profit)) >= allowed_dd:
                return f"champ_trail_rank{rank}"

        return None

    ###########################################################################
    # Helper: safe current profit ratio
    ###########################################################################
    def _current_profit_ratio(self, trade: Trade) -> Optional[float]:
        try:
            # Live ticker if available
            if self.dp:
                ticker = self.dp.ticker(trade.pair)
                if ticker and ticker.get("last") is not None:
                    rate = float(ticker["last"])
                    return float(trade.calc_profit_ratio(rate))
        except Exception:
            pass

        # Backtest/live fallback: last analyzed candle close
        rate2 = self._best_rate(trade.pair)
        if rate2 is not None:
            try:
                return float(trade.calc_profit_ratio(rate2))
            except Exception:
                return None

        return None

    ###########################################################################
    # Optional ROI logger (debug visibility)
    ###########################################################################
    def _roi_logger(self):
        while True:
            try:
                open_trades = Trade.get_open_trades()
                if not open_trades:
                    print("No open trades.")
                else:
                    now = datetime.utcnow().replace(tzinfo=timezone.utc)

                    for t in open_trades:
                        cp = self._current_profit_ratio(t)
                        if cp is None:
                            continue

                        side_trades = [x for x in open_trades if x.is_short == t.is_short]
                        ranks = self._power_ranks_side(side_trades)
                        rank = ranks.get(int(t.id), 1)

                        if t.is_short:
                            dyn = self._dynamic_roi_short(t, now)
                        else:
                            dyn = self._dynamic_roi_long(t, now)

                        if rank >= 4:
                            _, rung_target = self.gating_ladder.get(rank, (None, None))

                            if rung_target is not None:
                                dyn = max(
                                    dyn,
                                    dyn + 0.6 * (rung_target - dyn)
                                )

                        side = "SHORT" if t.is_short else "LONG"
                        print(
                            f"[{side} {t.pair}] "
                            f"rank={rank} profit={cp:.4f} dyn_roi={dyn:.4f}"
                        )

            except Exception as e:
                print(f"[ROI_LOGGER] error: {e}")

            time.sleep(60)

    def _side_key(self, is_short: bool) -> str:
        return "short" if is_short else "long"

    def _init_ignition_side(self, side_key: str) -> None:
        if side_key in self._ignition:
            return
        self._ignition[side_key] = {
            "baseline_time": None,        # datetime
            "baseline_pnl_abs": 0.0,      # absolute pnl at baseline snapshot
            "baseline_stake_abs": 1.0,    # deployed stake snapshot (avoid div0)
            "threshold": self.macro_ignition_threshold,
            "ever_had_3": False,
            "last_open_count": 0,
            "armed": False,               # baseline is set + slot3 gating active
        }

    def _trade_profit_abs_live(self, trade: Trade) -> float:
        """
        Approx abs PnL for an OPEN trade using best-effort current rate.
        Works in live + backtest.
        """
        rate = None

        # Live ticker if available
        try:
            if self.dp:
                ticker = self.dp.ticker(trade.pair)
                if ticker and ticker.get("last") is not None:
                    rate = float(ticker["last"])
        except Exception:
            pass

        # Backtest/live fallback: last analyzed candle close
        if rate is None:
            try:
                df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                if df is not None and not df.empty:
                    rate = float(df.iloc[-1]["close"])
            except Exception:
                pass

        if rate is None:
            return 0.0

        try:
            pr = float(trade.calc_profit_ratio(rate))
            stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
            return pr * stake
        except Exception:
            return 0.0

    def _closed_profit_abs(self, trade: Trade) -> float:
        """
        Best-effort absolute closed profit extraction across freqtrade versions.
        """
        for attr in ("close_profit_abs", "close_profit", "profit_abs", "profit"):
            try:
                v = getattr(trade, attr, None)
                if v is not None:
                    return float(v)
            except Exception:
                continue
        return 0.0

    def _get_closed_trades_best_effort(self) -> List[Trade]:
        """
        Try multiple APIs / fallbacks to get closed trades list.
        If nothing works, returns [] (slot-3 ignition will still work off open pnl).
        """
        # Some versions: Trade.get_trades(is_open=False)
        if hasattr(Trade, "get_trades"):
            try:
                return Trade.get_trades(is_open=False)  # type: ignore
            except TypeError:
                try:
                    return Trade.get_trades()  # type: ignore
                except Exception:
                    pass
            except Exception:
                pass

        # Some versions: Trade.get_closed_trades()
        if hasattr(Trade, "get_closed_trades"):
            try:
                return Trade.get_closed_trades()  # type: ignore
            except Exception:
                pass

        # SQLAlchemy fallback (if available)
        try:
            return Trade.query.filter(Trade.is_open.is_(False)).all()  # type: ignore
        except Exception:
            return []

    def _side_total_pnl_abs(self, is_short: bool, since_time: Optional[datetime] = None) -> float:
        """
        Total side pnl abs = realized(closed) + unrealized(open)
        Optionally only count closed trades after since_time; open is always live.
        """
        total = 0.0

        # unrealized (open)
        try:
            open_trades = Trade.get_open_trades()
            for t in open_trades:
                if bool(t.is_short) != bool(is_short):
                    continue
                total += self._trade_profit_abs_live(t)
        except Exception:
            pass

        # realized (closed)
        try:
            closed = self._get_closed_trades_best_effort()
            for t in closed:
                if bool(getattr(t, "is_short", False)) != bool(is_short):
                    continue
                if since_time is not None:
                    cd = getattr(t, "close_date", None)
                    if cd is None:
                        continue
                    # normalize tz
                    if cd.tzinfo is None:
                        cd = cd.replace(tzinfo=timezone.utc)
                    else:
                        cd = cd.astimezone(timezone.utc)

                    st = since_time
                    if st.tzinfo is None:
                        st = st.replace(tzinfo=timezone.utc)
                    else:
                        st = st.astimezone(timezone.utc)

                    if cd <= st:
                        continue

                total += self._closed_profit_abs(t)
        except Exception:
            pass

        return float(total)

    def _maybe_arm_or_rearm_slot3_ignition(self, is_short: bool, side_trades: List[Trade], now: datetime) -> None:
        """
        Arms ignition once the side has 2 open trades (baseline_time snapshot).
        Re-arms micro ignition when we had 3+ and fall back to 2.
        """
        side_key = self._side_key(is_short)
        self._init_ignition_side(side_key)
        st = self._ignition[side_key]

        count = len(side_trades)
        last = int(st.get("last_open_count", 0))
        st["last_open_count"] = count

        if count >= 3:
            st["ever_had_3"] = True

        nowu = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        if nowu.tzinfo:
            nowu = nowu.astimezone(timezone.utc)

        # MICRO re-arm: dropped from 3+ down to 2
        if bool(st.get("ever_had_3")) and last >= 3 and count == 2:
            baseline_stake = sum(float(getattr(t, "stake_amount", 0.0) or 0.0) for t in side_trades)
            st["baseline_time"] = nowu
            st["baseline_stake_abs"] = max(baseline_stake, 1.0)
            st["threshold"] = float(self.micro_ignition_threshold)
            st["armed"] = True
            return

        # MACRO arm: first time we reach exactly 2 open trades
        if count == 2 and st.get("baseline_time") is None:
            baseline_stake = sum(float(getattr(t, "stake_amount", 0.0) or 0.0) for t in side_trades)
            st["baseline_time"] = nowu
            st["baseline_stake_abs"] = max(baseline_stake, 1.0)
            st["threshold"] = float(self.macro_ignition_threshold)
            st["armed"] = True

    def _slot3_ignition_allows(self, is_short: bool, now: datetime) -> bool:
        """
        Allow slot3 only after side total PnL (closed+open) since baseline_time
        has moved by +/- threshold of baseline deployed stake snapshot.
        """
        side_key = self._side_key(is_short)
        self._init_ignition_side(side_key)
        st = self._ignition[side_key]

        if not bool(st.get("armed", False)) or st.get("baseline_time") is None:
            # If we can't prove we're armed, don't brick entries on restarts.
            return True

        baseline_time = st["baseline_time"]
        baseline_stake = float(st.get("baseline_stake_abs", 1.0))
        threshold = float(st.get("threshold", float(self.macro_ignition_threshold)))

        # PnL *since baseline_time* (this is the only value we need)
        pnl_since = self._side_total_pnl_abs(is_short=is_short, since_time=baseline_time)

        move_ratio = pnl_since / max(baseline_stake, 1.0)
        return abs(move_ratio) >= threshold

    def _best_rate(self, pair: str) -> Optional[float]:
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is not None and not df.empty:
                return float(df.iloc[-1]["close"])
        except Exception:
            pass
        return None
