from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame, Series
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from typing import Optional, Dict
from datetime import datetime


class Downforce(IStrategy):
    '''
    Downforce: full-lifecycle waveform recognition (rev.2)

    Thesis: some crypto moves carry a recognizable downforce signature - a
    waveform that recurs across several market phases before a high-conviction
    directional move. The edge is recognizing the full-pattern resemblance
    across the lifecycle and taking the earliest 1m trigger once the shape
    qualifies.

    Rev.2 adds strict vectorized lifecycle ordering (compression before expansion
    within the lookback window), pullback-quality and volume-conformity scoring,
    HTF RSI sanity filters, a stateful custom_exit organ, and a tiered
    custom_stoploss management genome. All edits are parametric and additive.
    '''

    timeframe = '1m'
    can_short = True
    process_only_new_candles = True
    use_exit_signal = False
    startup_candle_count = 600

    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    minimal_roi = {
        '0': 0.05,
        '30': 0.03,
        '90': 0.015,
        '240': 0.0,
    }

    # --- Tunable parameters -------------------------------------------------
    comp_floor = 0.40
    expand_floor = 0.35
    breakout_floor = 0.45
    confirm_floor = 0.45
    entry_threshold = 0.62
    exhaustion_exit = 0.62
    lifecycle_window = 120
    seq_expand_max_age = 40
    htf_rsi_long_max = 72.0
    htf_rsi_short_min = 28.0
    pullback_pct_max = 0.50
    vol_confirm_mult = 1.3

    _EPS = 1e-9

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._pair_analytics: Dict[str, dict] = {}

    def leverage(self, pair: str, current_time: datetime, rate: float,
                 proposed_leverage: float, **kwargs) -> float:
        return 1.0

    @staticmethod
    def _c01(series: Series, lo: float, hi: float) -> Series:
        return ((series - lo) / (hi - lo)).clip(lower=0.0, upper=1.0)

    # --- Higher-timeframe context -------------------------------------------
    @informative('5m')
    def populate_indicators_5m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)
        return dataframe

    @informative('15m')
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)
        return dataframe

    # --- ENTRY GENOME -------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        eps = self._EPS
        pair = metadata['pair']

        # Trend skeleton
        dataframe['ema_f'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_m'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_s'] = ta.EMA(dataframe, timeperiod=55)

        # Directional / strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=10)

        # Volatility envelope
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upper'] = bb['upper']
        dataframe['bb_lower'] = bb['lower']
        dataframe['bb_mid'] = bb['mid']
        dataframe['bbw'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / (dataframe['bb_mid'] + eps)

        # Range structure & participation
        dataframe['hh'] = dataframe['high'].rolling(20).max()
        dataframe['ll'] = dataframe['low'].rolling(20).min()
        dataframe['vol_ma'] = dataframe['volume'].rolling(30).mean()
        dataframe['vol_comp_ma'] = dataframe['volume'].rolling(60).mean()

        atr = dataframe['atr']
        bbw = dataframe['bbw']
        close = dataframe['close']
        high = dataframe['high']
        low = dataframe['low']

        # Stage 1: compression
        bbw_ratio = bbw / (bbw.rolling(120).median() + eps)
        atr_ratio = atr / (atr.rolling(120).median() + eps)
        dataframe['s_compression'] = (
            self._c01(1.3 - bbw_ratio, 0.0, 0.8) * self._c01(1.3 - atr_ratio, 0.0, 0.8)
        )

        # Stage 2: early expansion
        atr_mom = atr / (atr.shift(5) + eps) - 1.0
        dataframe['s_expansion'] = self._c01(atr_mom, 0.02, 0.30)

        # Stage 3: breakout (directional)
        prior_hh = dataframe['hh'].shift(1)
        prior_ll = dataframe['ll'].shift(1)
        vol_ratio = dataframe['volume'] / (dataframe['vol_ma'] + eps)
        vol_score = self._c01(vol_ratio, 1.0, 2.5)
        dataframe['s_breakout_long'] = (
            self._c01((close - prior_hh) / (atr + eps), 0.0, 1.0) * vol_score
        )
        dataframe['s_breakout_short'] = (
            self._c01((prior_ll - close) / (atr + eps), 0.0, 1.0) * vol_score
        )

        # Stage 4: trend confirmation (directional)
        adx_rising = dataframe['adx'] - dataframe['adx'].shift(3)
        ema_stack_long = ((dataframe['ema_f'] > dataframe['ema_m']) & (dataframe['ema_m'] > dataframe['ema_s'])).astype(float)
        ema_stack_short = ((dataframe['ema_f'] < dataframe['ema_m']) & (dataframe['ema_m'] < dataframe['ema_s'])).astype(float)
        di_spread = dataframe['plus_di'] - dataframe['minus_di']
        adx_quality = self._c01(dataframe['adx'], 18.0, 35.0)
        rising_score = self._c01(adx_rising, 0.0, 10.0)
        dataframe['s_confirm_long'] = (
            0.45 * ema_stack_long
            + 0.30 * self._c01(di_spread, 0.0, 25.0)
            + 0.15 * adx_quality
            + 0.10 * rising_score
        )
        dataframe['s_confirm_short'] = (
            0.45 * ema_stack_short
            + 0.30 * self._c01(-di_spread, 0.0, 25.0)
            + 0.15 * adx_quality
            + 0.10 * rising_score
        )

        # Stage 5: trend maturity (directional)
        mature = self._c01(dataframe['adx'], 22.0, 40.0)
        dataframe['s_maturity_long'] = mature * ema_stack_long
        dataframe['s_maturity_short'] = mature * ema_stack_short

        # Stage 6: exhaustion (against trade)
        ext = (close - dataframe['ema_s']) / (atr + eps)
        dataframe['s_exhaustion_long'] = (
            0.6 * self._c01(dataframe['rsi'], 72.0, 88.0) + 0.4 * self._c01(ext, 3.0, 9.0)
        )
        dataframe['s_exhaustion_short'] = (
            0.6 * self._c01(28.0 - dataframe['rsi'], 0.0, 16.0) + 0.4 * self._c01(-ext, 3.0, 9.0)
        )

        # --- Temporal sequence enforcement (vectorized lifecycle ordering) ---
        w = self.lifecycle_window
        comp_idx = dataframe['s_compression'].rolling(window=w, min_periods=w).apply(np.argmax, raw=True)
        expand_idx = dataframe['s_expansion'].rolling(window=w, min_periods=w).apply(np.argmax, raw=True)
        dataframe['seq_comp_before_expand'] = comp_idx < expand_idx
        dataframe['seq_expand_fresh'] = expand_idx >= (w - self.seq_expand_max_age)
        dataframe['comp_seen'] = dataframe['s_compression'].rolling(w).max()
        dataframe['expand_seen'] = dataframe['s_expansion'].rolling(w).max()

        # --- Pullback quality & volume conformity -----------------------------
        dataframe['local_high'] = dataframe['high'].rolling(10).max()
        dataframe['local_low'] = dataframe['low'].rolling(10).min()
        swing_range = dataframe['local_high'] - dataframe['local_low']
        pullback_long = (dataframe['local_high'] - low) / (swing_range + eps)
        pullback_short = (high - dataframe['local_low']) / (swing_range + eps)
        dataframe['pullback_long'] = 1.0 - self._c01(pullback_long, 0.0, self.pullback_pct_max)
        dataframe['pullback_short'] = 1.0 - self._c01(pullback_short, 0.0, self.pullback_pct_max)

        vol_conform = dataframe['volume'] / (dataframe['vol_comp_ma'] + eps)
        dataframe['vol_conform'] = self._c01(vol_conform, self.vol_confirm_mult, 3.0)

        # --- Composite Downforce similarity ----------------------------------
        dataframe['df_long'] = (
            0.16 * dataframe['comp_seen']
            + 0.12 * dataframe['expand_seen']
            + 0.24 * dataframe['s_breakout_long']
            + 0.24 * dataframe['s_confirm_long']
            + 0.14 * dataframe['s_maturity_long']
            + 0.06 * dataframe['pullback_long']
            + 0.04 * dataframe['vol_conform']
        ).fillna(0.0)
        dataframe['df_short'] = (
            0.16 * dataframe['comp_seen']
            + 0.12 * dataframe['expand_seen']
            + 0.24 * dataframe['s_breakout_short']
            + 0.24 * dataframe['s_confirm_short']
            + 0.14 * dataframe['s_maturity_short']
            + 0.06 * dataframe['pullback_short']
            + 0.04 * dataframe['vol_conform']
        ).fillna(0.0)

        # Cache latest analytics for management genome
        if not dataframe.empty:
            last = dataframe.iloc[-1]
            self._pair_analytics[pair] = {
                's_exhaustion_long': float(last['s_exhaustion_long']),
                's_exhaustion_short': float(last['s_exhaustion_short']),
                's_maturity_long': float(last['s_maturity_long']),
                's_maturity_short': float(last['s_maturity_short']),
                's_confirm_long': float(last['s_confirm_long']),
                's_confirm_short': float(last['s_confirm_short']),
                'adx': float(last['adx']),
            }

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        htf_long = (
            (dataframe['ema_fast_15m'] > dataframe['ema_slow_15m'])
            & (dataframe['ema_fast_1h'] > dataframe['ema_slow_1h'])
            & (dataframe['rsi_15m'].fillna(50) < self.htf_rsi_long_max)
        )
        htf_short = (
            (dataframe['ema_fast_15m'] < dataframe['ema_slow_15m'])
            & (dataframe['ema_fast_1h'] < dataframe['ema_slow_1h'])
            & (dataframe['rsi_15m'].fillna(50) > self.htf_rsi_short_min)
        )

        qualify_long = (
            (dataframe['comp_seen'] >= self.comp_floor)
            & (dataframe['expand_seen'] >= self.expand_floor)
            & (dataframe['seq_comp_before_expand'].fillna(False))
            & (dataframe['seq_expand_fresh'].fillna(False))
            & (dataframe['s_breakout_long'] >= self.breakout_floor)
            & (dataframe['s_confirm_long'] >= self.confirm_floor)
            & htf_long
            & (dataframe['df_long'] >= self.entry_threshold)
        ).fillna(False)

        qualify_short = (
            (dataframe['comp_seen'] >= self.comp_floor)
            & (dataframe['expand_seen'] >= self.expand_floor)
            & (dataframe['seq_comp_before_expand'].fillna(False))
            & (dataframe['seq_expand_fresh'].fillna(False))
            & (dataframe['s_breakout_short'] >= self.breakout_floor)
            & (dataframe['s_confirm_short'] >= self.confirm_floor)
            & htf_short
            & (dataframe['df_short'] >= self.entry_threshold)
        ).fillna(False)

        enter_long = qualify_long & ~qualify_long.shift(1).fillna(False)
        enter_short = qualify_short & ~qualify_short.shift(1).fillna(False)

        tag_long = 'DFL:' + (dataframe['df_long'] * 100).round().fillna(0).astype(int).astype(str)
        tag_short = 'DFS:' + (dataframe['df_short'] * 100).round().fillna(0).astype(int).astype(str)

        dataframe.loc[enter_long, 'enter_long'] = 1
        dataframe.loc[enter_long, 'enter_tag'] = tag_long
        dataframe.loc[enter_short, 'enter_short'] = 1
        dataframe.loc[enter_short, 'enter_tag'] = tag_short

        return dataframe

    # --- EXIT GENOME --------------------------------------------------------
    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float,
                    current_profit: float, current_profit_ratio: float, dataframe: DataFrame,
                    metadata: dict, **kwargs) -> Optional[str]:
        if dataframe.empty:
            return None
        last = dataframe.iloc[-1]
        is_long = not trade.is_short
        dir_suff = 'long' if is_long else 'short'

        # Exhaustion exit
        exh_key = f's_exhaustion_{dir_suff}'
        if last[exh_key] >= self.exhaustion_exit and current_profit_ratio > 0.008:
            return f'df_exhaustion_{dir_suff}'

        # Invalidation: confirmation collapses + ADX crumbling
        conf_key = f's_confirm_{dir_suff}'
        confirm_collapse = last[conf_key] < 0.25
        adx_crumble = (
            last['adx'] < 20.0
            and (last['adx'] - dataframe['adx'].shift(3).iloc[-1]) < -6.0
        )
        if confirm_collapse and adx_crumble:
            return f'df_invalidation_{dir_suff}'

        # Reversal: structural DI flip + HTF disagreement
        if is_long:
            rev = (
                qtpylib.crossed_below(dataframe['ema_f'], dataframe['ema_m']).iloc[-1]
                or (last['minus_di'] > last['plus_di'])
            )
            htf_rev = last.get('ema_fast_1h', 0) < last.get('ema_slow_1h', 1)
        else:
            rev = (
                qtpylib.crossed_above(dataframe['ema_f'], dataframe['ema_m']).iloc[-1]
                or (last['plus_di'] > last['minus_di'])
            )
            htf_rev = last.get('ema_fast_1h', 1) > last.get('ema_slow_1h', 0)

        if rev and htf_rev:
            return f'df_reversal_{dir_suff}'

        return None

    # --- MANAGEMENT GENOME --------------------------------------------------
    def custom_stoploss(self, pair: str, trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_high_profit: float, **kwargs) -> Optional[float]:
        analytics = self._pair_analytics.get(pair, {})
        if not analytics:
            return None

        is_long = not trade.is_short
        sign = -1.0 if is_long else 1.0
        dir_suff = 'long' if is_long else 'short'

        # Breathing room while profit is low
        if current_profit < 0.015:
            return sign * 0.035

        # Profit protection zone
        if current_profit < 0.04:
            return sign * 0.015

        # Maturity-based tightening
        maturity = analytics.get(f's_maturity_{dir_suff}', 0.0)
        if maturity > 0.55:
            return sign * 0.008

        # Exhaustion-driven emergency tighten
        exhaustion = analytics.get(f's_exhaustion_{dir_suff}', 0.0)
        if exhaustion > 0.45:
            return sign * 0.005

        return None
