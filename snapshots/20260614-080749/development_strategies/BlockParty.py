from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import timedelta


class BlockParty(IStrategy):
    '''
    Block Party (rev.5): cartographer laggard-catch-up strategy.
    Revision focus: narrow parametric tweak per B-grade review (79.0, 1 trade/6h).
    Loosened group_move_min, laggard_gap, min_breadth, and vol_mult slightly
    to gently improve opportunity capture without disturbing the three-genome
    architecture or the working exit/management layers.
    '''

    timeframe = '15m'
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 160

    stoploss = -0.07
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.035
    trailing_only_offset_is_reached = True

    minimal_roi = {
        '0': 0.06,
        '60': 0.03,
        '180': 0.015,
        '480': 0.0,
    }

    # --- Tunable parameters (narrow tweak rev.5) -----------------------------
    lookback = 96               # 24h on 15m bars (slow constellation)
    lookback_fast = 16          # 4h on 15m bars (early expansion track)
    min_group_pct = 0.30
    group_move_min = 0.8        # loosened: catch smaller lit moves
    group_move_fast_min = 0.5   # fast-window lit threshold lowered
    group_move_strong = 3.0     # strong move threshold
    laggard_gap = 0.6           # lowered core gap
    soft_laggard_gap = 0.15     # lowered soft gap
    deep_laggard_gap = 1.5      # deep-value laggard track
    min_breadth = 0.25          # slightly relaxed coherence
    group_exit_level = 0.4      # cool-off threshold
    rsi_overbought = 80
    rsi_oversold = 20
    vol_mult = 0.15             # volume floor relaxed modestly
    early_catchup_pct = 0.12    # bar-move ignition threshold relaxed
    catchup_exit = 0.4          # profit-take when gap closes
    custom_time_stop_min = 90
    custom_time_stop_dd = -0.005
    bb_period = 20
    bb_std = 2.0
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._group_cache_key = None
        self._group_cache_val = (None, None, None, None, None)

    def leverage(self, pair: str, current_time, rate: float,
                 proposed_leverage: float, **kwargs) -> float:
        return 1.0

    # -------------------------------------------------------------------------
    # --- MANAGEMENT GENOME (position sizing / DCA) ---------------------------
    # -------------------------------------------------------------------------
    def adjust_trade_position(self, trade, current_time, current_rate,
                              current_profit, min_stake, max_stake,
                              current_entry_rate, current_exit_rate,
                              current_entry_profit, **kwargs):
        '''
        Management genome: Block Party is concentrated and thesis-driven.
        No DCA / pyramid scaling is applied; conviction is set at entry.
        '''
        return None

    # -------------------------------------------------------------------------
    # --- ENTRY GENOME (cross-sectional opportunity detection) ----------------
    # -------------------------------------------------------------------------
    def _group_stats(self, current_df: DataFrame):
        '''
        Compute median return, breadth, and count across the current
        served constellation for both slow and fast lookbacks.
        Cached per candle timestamp.
        '''
        if not self.dp:
            return None, None, None, None, None

        try:
            pairs = self.dp.current_whitelist()
        except Exception:
            return None, None, None, None, None

        if not pairs:
            return None, None, None, None, None

        try:
            cache_key = str(current_df['date'].iloc[-1])
        except Exception:
            cache_key = None

        if cache_key and self._group_cache_key == cache_key:
            return self._group_cache_val

        rets_slow = []
        rets_fast = []
        for p in pairs:
            try:
                df, _ = self.dp.get_analyzed_dataframe(p, self.timeframe)
                if df is None or len(df) < self.lookback + 2:
                    continue
                close = df['close']
                now = close.iloc[-1]
                if now is None or not np.isfinite(now):
                    continue

                past_slow = close.iloc[-1 - self.lookback]
                if past_slow is not None and past_slow > 0 and np.isfinite(past_slow):
                    ret_slow = (now / past_slow - 1.0) * 100.0
                    if np.isfinite(ret_slow):
                        rets_slow.append(float(ret_slow))

                past_fast = close.iloc[-1 - self.lookback_fast]
                if past_fast is not None and past_fast > 0 and np.isfinite(past_fast):
                    ret_fast = (now / past_fast - 1.0) * 100.0
                    if np.isfinite(ret_fast):
                        rets_fast.append(float(ret_fast))
            except Exception:
                continue

        min_needed = max(2, int(len(pairs) * self.min_group_pct))
        if len(rets_slow) < min_needed:
            return None, None, None, None, None

        arr_slow = np.array(rets_slow)
        median_slow = float(np.median(arr_slow))
        if abs(median_slow) < 0.05:
            median_slow = 0.0

        aligned_slow = 0
        for r in rets_slow:
            if median_slow > 0 and r > 0:
                aligned_slow += 1
            elif median_slow < 0 and r < 0:
                aligned_slow += 1
            elif abs(median_slow) < 0.1:
                aligned_slow += 1
        breadth_slow = aligned_slow / len(rets_slow) if rets_slow else 0.0

        median_fast = None
        breadth_fast = None
        n_fast = len(rets_fast)
        if n_fast >= min_needed:
            arr_fast = np.array(rets_fast)
            median_fast = float(np.median(arr_fast))
            if abs(median_fast) < 0.05:
                median_fast = 0.0
            aligned_fast = 0
            for r in rets_fast:
                if median_fast > 0 and r > 0:
                    aligned_fast += 1
                elif median_fast < 0 and r < 0:
                    aligned_fast += 1
                elif abs(median_fast) < 0.1:
                    aligned_fast += 1
            breadth_fast = aligned_fast / n_fast if n_fast else 0.0

        val = (median_slow, breadth_slow, len(rets_slow), median_fast, breadth_fast)
        if cache_key:
            self._group_cache_key = cache_key
            self._group_cache_val = val
        return val

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Window returns
        dataframe['ret_window'] = dataframe['close'].pct_change(self.lookback) * 100.0
        dataframe['ret_window_fast'] = dataframe['close'].pct_change(self.lookback_fast) * 100.0
        dataframe['ret_current'] = dataframe['close'].pct_change(1) * 100.0

        # Trend & momentum
        dataframe['ema_f'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_m'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_s'] = ta.EMA(dataframe, timeperiod=55)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['vol_ma'] = dataframe['volume'].rolling(20).mean()
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # Bollinger Bands
        bbands = ta.BBANDS(
            dataframe,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
        )
        dataframe['lower'] = bbands['lowerband']
        dataframe['middle'] = bbands['middleband']
        dataframe['upper'] = bbands['upperband']

        # MACD
        macd = ta.MACD(
            dataframe,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Broadened soft confirmation booleans (additive depth / loosened)
        dataframe['sig_catchup_long'] = dataframe['ret_current'] >= self.early_catchup_pct
        dataframe['sig_catchup_short'] = dataframe['ret_current'] <= -self.early_catchup_pct

        # Simplified EMA: price vs medium EMA only
        dataframe['sig_ema_long'] = dataframe['close'] > dataframe['ema_m']
        dataframe['sig_ema_short'] = dataframe['close'] < dataframe['ema_m']

        # BB touch only (direction of candle removed)
        dataframe['sig_bb_long'] = dataframe['close'] <= dataframe['lower']
        dataframe['sig_bb_short'] = dataframe['close'] >= dataframe['upper']

        # MACD histogram sign only (momentum direction, not acceleration)
        dataframe['sig_macd_long'] = dataframe['macdhist'] > 0
        dataframe['sig_macd_short'] = dataframe['macdhist'] < 0

        # RSI turning (level caps removed)
        dataframe['sig_rsi_long'] = dataframe['rsi'] > dataframe['rsi'].shift(1)
        dataframe['sig_rsi_short'] = dataframe['rsi'] < dataframe['rsi'].shift(1)

        # Extremity guardrails
        dataframe['sig_not_extreme_long'] = dataframe['rsi'] < self.rsi_overbought
        dataframe['sig_not_extreme_short'] = dataframe['rsi'] > self.rsi_oversold

        # Composite scores
        dataframe['score_long'] = (
            dataframe['sig_catchup_long'].fillna(False).astype(int) +
            dataframe['sig_ema_long'].fillna(False).astype(int) +
            dataframe['sig_bb_long'].fillna(False).astype(int) +
            dataframe['sig_macd_long'].fillna(False).astype(int) +
            dataframe['sig_rsi_long'].fillna(False).astype(int)
        )

        dataframe['score_short'] = (
            dataframe['sig_catchup_short'].fillna(False).astype(int) +
            dataframe['sig_ema_short'].fillna(False).astype(int) +
            dataframe['sig_bb_short'].fillna(False).astype(int) +
            dataframe['sig_macd_short'].fillna(False).astype(int) +
            dataframe['sig_rsi_short'].fillna(False).astype(int)
        )

        dataframe['vol_ok'] = dataframe['volume'] > (dataframe['vol_ma'] * self.vol_mult)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Explicit column init to guarantee clean state
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''

        group, breadth, n, group_fast, breadth_fast = self._group_stats(dataframe)
        if group is None or breadth is None:
            return dataframe

        # At least one lookback must show coherence
        breadth_ok = (breadth >= self.min_breadth) or (breadth_fast is not None and breadth_fast >= self.min_breadth)
        if not breadth_ok:
            return dataframe

        # Lit constellation checks (slow OR fast)
        lit_up = (group >= self.group_move_min) or (group_fast is not None and group_fast >= self.group_move_fast_min)
        lit_down = (group <= -self.group_move_min) or (group_fast is not None and group_fast <= -self.group_move_fast_min)
        strong_up = (group >= self.group_move_strong) or (group_fast is not None and group_fast >= self.group_move_strong)
        strong_down = (group <= -self.group_move_strong) or (group_fast is not None and group_fast <= -self.group_move_strong)

        own = dataframe['ret_window']
        gap = own - group

        # --- LONG: constellation pumping ---
        if lit_up:
            is_laggard = gap <= -self.laggard_gap
            is_soft_laggard = strong_up & (gap <= -self.soft_laggard_gap)
            is_deep = gap <= -self.deep_laggard_gap

            # Track A: confirmed laggard with at least one soft signal
            cond_core = (
                (is_laggard | is_soft_laggard) &
                (dataframe['score_long'] >= 1) &
                dataframe['vol_ok'] &
                dataframe['sig_not_extreme_long']
            )

            # Track B: deep laggard with minimal RSI stabilization
            cond_deep = (
                is_deep &
                dataframe['vol_ok'] &
                dataframe['sig_not_extreme_long'] &
                dataframe['sig_rsi_long']
            )

            # Track C: ignition turn (simplified EMA + RSI turn + positive bar)
            cond_ignition = (
                is_laggard &
                dataframe['vol_ok'] &
                dataframe['sig_ema_long'] &
                dataframe['sig_rsi_long'] &
                (dataframe['ret_current'] > 0)
            )

            enter = cond_core | cond_deep | cond_ignition
            enter = enter & (~enter.shift(1).fillna(False))
            gf_str = f'{group_fast:+.1f}' if group_fast is not None else 'na'
            tag = f'BPL_g{group:+.1f}_gf{gf_str}_brd{int(breadth*100)}_n{n}'
            dataframe.loc[enter, 'enter_long'] = 1
            dataframe.loc[enter, 'enter_tag'] = tag

        # --- SHORT: constellation dumping ---
        elif lit_down:
            is_laggard = gap >= self.laggard_gap
            is_soft_laggard = strong_down & (gap >= self.soft_laggard_gap)
            is_deep = gap >= self.deep_laggard_gap

            cond_core = (
                (is_laggard | is_soft_laggard) &
                (dataframe['score_short'] >= 1) &
                dataframe['vol_ok'] &
                dataframe['sig_not_extreme_short']
            )

            cond_deep = (
                is_deep &
                dataframe['vol_ok'] &
                dataframe['sig_not_extreme_short'] &
                dataframe['sig_rsi_short']
            )

            cond_ignition = (
                is_laggard &
                dataframe['vol_ok'] &
                dataframe['sig_ema_short'] &
                dataframe['sig_rsi_short'] &
                (dataframe['ret_current'] < 0)
            )

            enter = cond_core | cond_deep | cond_ignition
            enter = enter & (~enter.shift(1).fillna(False))
            gf_str = f'{group_fast:+.1f}' if group_fast is not None else 'na'
            tag = f'BPS_g{group:+.1f}_gf{gf_str}_brd{int(breadth*100)}_n{n}'
            dataframe.loc[enter, 'enter_short'] = 1
            dataframe.loc[enter, 'enter_tag'] = tag

        return dataframe

    # -------------------------------------------------------------------------
    # --- EXIT GENOME (exit logic) --------------------------------------------
    # -------------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Explicit column init
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''

        group, _, _, _, _ = self._group_stats(dataframe)
        own = dataframe['ret_window']
        gap = (own - group) if group is not None else None

        # Technical reversal triggers (hardened NaN handling)
        cross_down = qtpylib.crossed_below(dataframe['ema_f'], dataframe['ema_m']).fillna(False)
        rsi_high = (dataframe['rsi'] > self.rsi_overbought).fillna(False)
        rev_long = cross_down | rsi_high

        cross_up = qtpylib.crossed_above(dataframe['ema_f'], dataframe['ema_m']).fillna(False)
        rsi_low = (dataframe['rsi'] < self.rsi_oversold).fillna(False)
        rev_short = cross_up | rsi_low

        exit_long = rev_long
        exit_short = rev_short

        if group is not None and gap is not None:
            exit_long = exit_long | (gap.fillna(-9999) >= -self.catchup_exit)
            exit_short = exit_short | (gap.fillna(9999) <= self.catchup_exit)
            exit_long = exit_long | (group < self.group_exit_level)
            exit_short = exit_short | (group > -self.group_exit_level)

        # Bollinger extreme rejection
        bb_reject_long = (
            (dataframe['close'] >= dataframe['upper']) & (dataframe['open'] < dataframe['close'])
        ).fillna(False)
        bb_reject_short = (
            (dataframe['close'] <= dataframe['lower']) & (dataframe['open'] > dataframe['close'])
        ).fillna(False)

        exit_long = exit_long | bb_reject_long
        exit_short = exit_short | bb_reject_short

        dataframe.loc[exit_long, 'exit_long'] = 1
        dataframe.loc[exit_long, 'exit_tag'] = 'bp_tech_long'
        dataframe.loc[exit_short, 'exit_short'] = 1
        dataframe.loc[exit_short, 'exit_tag'] = 'bp_tech_short'

        return dataframe

    def custom_exit(self, pair: str, trade, current_time, current_rate,
                    current_profit, **kwargs):
        '''
        Custom exit layer: fine-grained trade management once in a position.
        - Time stop: if the laggard hasn't moved after N minutes, cut the trade.
        - Leadership flip: if the former laggard is now outperforming the group,
          the edge is exhausted; take profit even if technicals haven't fired.
        '''
        # 1) Time-based discouragement stop
        if trade.open_date_utc is not None:
            open_date = trade.open_date_utc
            if open_date.tzinfo is None and current_time.tzinfo is not None:
                open_date = open_date.replace(tzinfo=current_time.tzinfo)
            elif open_date.tzinfo is not None and current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=open_date.tzinfo)
            duration = current_time - open_date
            if duration >= timedelta(minutes=self.custom_time_stop_min):
                if current_profit <= self.custom_time_stop_dd:
                    return 'bp_time_stop'

        # 2) Leadership / gap-reversal check (live read of latest group stats)
        if not self.dp:
            return None

        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is None or len(df) < self.lookback + 2:
                return None
            group, _, _, _, _ = self._group_stats(df)
            if group is None:
                return None
            close = df['close']
            past = close.iloc[-1 - self.lookback]
            now = close.iloc[-1]
            if past is None or past <= 0:
                return None
            own_ret = (now / past - 1.0) * 100.0
            gap = own_ret - group

            # If we caught up and now lead the pack, edge is gone
            if not trade.is_short and gap > 0.3:
                return 'bp_caught_up_long'
            if trade.is_short and gap < -0.3:
                return 'bp_caught_up_short'

            # If the group reversed against us dramatically, bail
            if not trade.is_short and group < -0.5:
                return 'bp_group_rev_long'
            if trade.is_short and group > 0.5:
                return 'bp_group_rev_short'

        except Exception:
            return None

        return None
