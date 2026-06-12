from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import timedelta


class BlockParty(IStrategy):
    '''
    Block Party (rev.3): the cartographer's laggard-catch-up strategy.
    Deepened revision: looser entry gates so the thesis actually fires,
    additive alt-entry path, hardened cross-sectional plumbing,
    dual-layer exit (technical + custom trade-duration/gap stop),
    and clean organ-separated genome layout.
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

    # --- Tunable parameters (loosened per shift review) --------------------
    lookback = 96               # 24h on 15m bars
    min_group_pct = 0.30        # lowered: need only 30% of whitelist with data
    group_move_min = 1.5        # lowered: 1.5% median move counts as lit
    group_move_strong = 3.5     # strong move threshold for soft-laggard mode
    laggard_gap = 1.0           # lowered: 1% trail is enough
    soft_laggard_gap = 0.3      # for strong constellations, enter even slight laggards
    min_breadth = 0.35          # lowered coherence requirement
    group_exit_level = 0.5      # cool-off threshold
    rsi_overbought = 78
    rsi_oversold = 22
    vol_mult = 0.25             # lowered volume floor
    early_catchup_pct = 0.25    # bar move that counts as ignition
    catchup_exit = 0.5          # take profits when gap closes to 0.5%
    custom_time_stop_min = 90   # minutes
    custom_time_stop_dd = -0.005  # -0.5% time-stop loss
    bb_period = 20
    bb_std = 2.0
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._group_cache_key = None
        self._group_cache_val = (None, None, None)

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
        Compute median return, breadth ratio, and count across the current
        served constellation (RemotePairList whitelist). Cached per candle.
        '''
        if not self.dp:
            return None, None, None

        try:
            pairs = self.dp.current_whitelist()
        except Exception:
            return None, None, None

        if not pairs:
            return None, None, None

        try:
            cache_key = str(current_df['date'].iloc[-1])
        except Exception:
            cache_key = None

        if cache_key and self._group_cache_key == cache_key:
            return self._group_cache_val

        rets = []
        for p in pairs:
            try:
                df, _ = self.dp.get_analyzed_dataframe(p, self.timeframe)
                if df is None or len(df) < self.lookback + 2:
                    continue
                close = df['close']
                past = close.iloc[-1 - self.lookback]
                now = close.iloc[-1]
                if past is None or now is None or past <= 0:
                    continue
                if not np.isfinite(past) or not np.isfinite(now):
                    continue
                ret = (now / past - 1.0) * 100.0
                if np.isfinite(ret):
                    rets.append(float(ret))
            except Exception:
                continue

        min_needed = max(2, int(len(pairs) * self.min_group_pct))
        if len(rets) < min_needed:
            return None, None, None

        arr = np.array(rets)
        median_ret = float(np.median(arr))
        # Snap tiny noise to zero to avoid phantom direction
        if abs(median_ret) < 0.05:
            median_ret = 0.0

        aligned = 0
        for r in rets:
            if median_ret > 0 and r > 0:
                aligned += 1
            elif median_ret < 0 and r < 0:
                aligned += 1
            elif abs(median_ret) < 0.1:
                aligned += 1
        breadth = aligned / len(rets) if rets else 0.0

        val = (median_ret, breadth, len(rets))
        if cache_key:
            self._group_cache_key = cache_key
            self._group_cache_val = val
        return val

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Window returns
        dataframe['ret_window'] = dataframe['close'].pct_change(self.lookback) * 100.0
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

        # Soft confirmation booleans (additive depth)
        dataframe['sig_catchup_long'] = dataframe['ret_current'] >= self.early_catchup_pct
        dataframe['sig_catchup_short'] = dataframe['ret_current'] <= -self.early_catchup_pct

        dataframe['sig_ema_long'] = (
            (dataframe['close'] > dataframe['ema_f']) & (dataframe['ema_f'] > dataframe['ema_m'])
        )
        dataframe['sig_ema_short'] = (
            (dataframe['close'] < dataframe['ema_f']) & (dataframe['ema_f'] < dataframe['ema_m'])
        )

        dataframe['sig_bb_long'] = (
            (dataframe['close'] <= dataframe['lower']) & (dataframe['close'] > dataframe['open'])
        )
        dataframe['sig_bb_short'] = (
            (dataframe['close'] >= dataframe['upper']) & (dataframe['close'] < dataframe['open'])
        )

        dataframe['sig_macd_long'] = (
            (dataframe['macdhist'] > 0) & (dataframe['macdhist'] > dataframe['macdhist'].shift(1))
        )
        dataframe['sig_macd_short'] = (
            (dataframe['macdhist'] < 0) & (dataframe['macdhist'] < dataframe['macdhist'].shift(1))
        )

        dataframe['sig_rsi_long'] = (
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) & (dataframe['rsi'] < 45)
        )
        dataframe['sig_rsi_short'] = (
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) & (dataframe['rsi'] > 55)
        )

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

        group, breadth, n = self._group_stats(dataframe)
        if group is None or breadth is None:
            return dataframe

        if breadth < self.min_breadth:
            return dataframe

        own = dataframe['ret_window']
        gap = own - group
        lit_up = group >= self.group_move_min
        lit_down = group <= -self.group_move_min
        strong_up = group >= self.group_move_strong
        strong_down = group <= -self.group_move_strong

        # --- LONG: constellation pumping ---
        if lit_up:
            is_laggard = gap <= -self.laggard_gap
            is_soft_laggard = strong_up & (gap <= -self.soft_laggard_gap)
            is_igniting = (
                (dataframe['close'] > dataframe['ema_f']) &
                (dataframe['rsi'] < 45) &
                (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
                (dataframe['ret_current'] > 0)
            )

            cond_core = (is_laggard | is_soft_laggard) & (dataframe['score_long'] >= 1) & dataframe['vol_ok']
            cond_alt = is_laggard & is_igniting & dataframe['vol_ok']

            enter = cond_core | cond_alt
            enter = enter & (~enter.shift(1).fillna(False))
            dataframe.loc[enter, 'enter_long'] = 1
            dataframe.loc[enter, 'enter_tag'] = f'BPL_g{group:+.1f}_brd{int(breadth*100)}_n{n}'

        # --- SHORT: constellation dumping ---
        elif lit_down:
            is_laggard = gap >= self.laggard_gap
            is_soft_laggard = strong_down & (gap >= self.soft_laggard_gap)
            is_igniting = (
                (dataframe['close'] < dataframe['ema_f']) &
                (dataframe['rsi'] > 55) &
                (dataframe['rsi'] < dataframe['rsi'].shift(1)) &
                (dataframe['ret_current'] < 0)
            )

            cond_core = (is_laggard | is_soft_laggard) & (dataframe['score_short'] >= 1) & dataframe['vol_ok']
            cond_alt = is_laggard & is_igniting & dataframe['vol_ok']

            enter = cond_core | cond_alt
            enter = enter & (~enter.shift(1).fillna(False))
            dataframe.loc[enter, 'enter_short'] = 1
            dataframe.loc[enter, 'enter_tag'] = f'BPS_g{group:+.1f}_brd{int(breadth*100)}_n{n}'

        return dataframe

    # -------------------------------------------------------------------------
    # --- EXIT GENOME (exit logic) --------------------------------------------
    # -------------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Explicit column init
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''

        group, _, _ = self._group_stats(dataframe)
        own = dataframe['ret_window']
        gap = (own - group) if group is not None else None

        # Technical reversal triggers
        rev_long = (
            qtpylib.crossed_below(dataframe['ema_f'], dataframe['ema_m']) |
            (dataframe['rsi'] > self.rsi_overbought)
        )
        rev_short = (
            qtpylib.crossed_above(dataframe['ema_f'], dataframe['ema_m']) |
            (dataframe['rsi'] < self.rsi_oversold)
        )

        exit_long = rev_long.fillna(False)
        exit_short = rev_short.fillna(False)

        if group is not None and gap is not None:
            # Caught-up exit: laggard closed the gap
            exit_long = exit_long | (gap >= -self.catchup_exit)
            exit_short = exit_short | (gap <= self.catchup_exit)
            # Constellation cooled off: the whole group move evaporated
            exit_long = exit_long | (group < self.group_exit_level)
            exit_short = exit_short | (group > -self.group_exit_level)

        # Bollinger extreme rejection (price hitting opposite band)
        bb_reject_long = (
            (dataframe['close'] >= dataframe['upper']) &
            (dataframe['open'] < dataframe['close'])
        )
        bb_reject_short = (
            (dataframe['close'] <= dataframe['lower']) &
            (dataframe['open'] > dataframe['close'])
        )

        exit_long = exit_long | bb_reject_long
        exit_short = exit_short | bb_reject_short

        exit_long = exit_long.fillna(False)
        exit_short = exit_short.fillna(False)

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
            group, _, _ = self._group_stats(df)
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
