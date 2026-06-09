from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class BlockParty(IStrategy):
    """
    Block Party (rev.2): the cartographer's laggard-catch-up strategy.

    Thesis: a served constellation (custom RemotePairList) lights up with a
    coherent directional move. Instead of chasing the leaders, we wait for the
    weakest members that still belong, then enter as they begin to turn toward
    the group.  This revision keeps the group-median scaffolding but replaces
    the strict EMA+RSI trend gate with a composite soft-confirmation score
    so the strategy can actually express its brief when the neighborhood is hot.
    """

    timeframe = "15m"
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
        "0": 0.06,
        "60": 0.03,
        "180": 0.015,
        "480": 0.0,
    }

    # --- Tunable parameters -------------------------------------------------
    lookback = 96               # 24h on 15m bars for the cross-sectional window
    min_group_pct = 0.4         # need >= 40% of whitelist with data (loosened)
    group_move_min = 2.0        # % median move for the constellation to count as lit
    laggard_gap = 2.5           # % a member must trail the group (loosened from 3.0)
    group_exit_level = 0.5      # constellation cooled below this median move
    rsi_overbought = 78
    rsi_oversold = 22
    vol_mult = 0.4              # volume floor loosened from 0.5x
    early_catchup_pct = 0.4     # current-bar move that counts as ignition
    catchup_exit = 0.8          # gap closed within this -> take profits
    bb_period = 20
    bb_std = 2.0
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._group_cache = {}

    def leverage(self, pair: str, current_time, rate: float,
                 proposed_leverage: float, **kwargs) -> float:
        return 1.0

    def _group_stats(self):
        """Return (median_return, breadth_ratio, count) for the current whitelist."""
        if not self.dp:
            return None, None, None
        try:
            pairs = self.dp.current_whitelist()
        except Exception:
            return None, None, None
        if not pairs:
            return None, None, None

        rets = []
        cache_key = None
        for pair in pairs:
            try:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            except Exception:
                continue
            if df is None or len(df) < self.lookback + 1:
                continue
            close = df["close"]
            past = close.iloc[-1 - self.lookback]
            if past is None or not np.isfinite(past) or past <= 0:
                continue
            ret = (close.iloc[-1] / past - 1.0) * 100.0
            if np.isfinite(ret):
                rets.append(float(ret))
            if cache_key is None:
                cache_key = str(df["date"].iloc[-1])

        min_needed = max(3, int(len(pairs) * self.min_group_pct))
        if len(rets) < min_needed:
            return None, None, None

        median_ret = float(np.median(rets))
        aligned = sum(
            1 for r in rets
            if (median_ret > 0 and r > 0)
            or (median_ret < 0 and r < 0)
            or abs(median_ret) < 0.1
        )
        breadth = aligned / len(rets)
        if cache_key is not None:
            self._group_cache[cache_key] = (median_ret, breadth, len(rets))
        return median_ret, breadth, len(rets)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Window returns
        dataframe["ret_window"] = dataframe["close"].pct_change(self.lookback) * 100.0
        dataframe["ret_current"] = dataframe["close"].pct_change(1) * 100.0

        # Trend & momentum
        dataframe["ema_f"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_m"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_s"] = ta.EMA(dataframe, timeperiod=55)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["vol_ma"] = dataframe["volume"].rolling(20).mean()

        # Bollinger Bands for extreme-laggard bounce detection
        bbands = ta.BBANDS(
            dataframe,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
        )
        dataframe["lower"] = bbands["lowerband"]
        dataframe["middle"] = bbands["middleband"]
        dataframe["upper"] = bbands["upperband"]

        # MACD for early-turn confirmation
        macd = ta.MACD(
            dataframe,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # Soft confirmation signals (looser than the old hard EMA+RSI gate)
        dataframe["sig_catchup_long"] = dataframe["ret_current"] >= self.early_catchup_pct
        dataframe["sig_catchup_short"] = dataframe["ret_current"] <= -self.early_catchup_pct

        dataframe["sig_ema_long"] = (
            (dataframe["close"] > dataframe["ema_f"]) & (dataframe["ema_f"] > dataframe["ema_m"])
        )
        dataframe["sig_ema_short"] = (
            (dataframe["close"] < dataframe["ema_f"]) & (dataframe["ema_f"] < dataframe["ema_m"])
        )

        dataframe["sig_bb_long"] = (
            (dataframe["close"] <= dataframe["lower"]) & (dataframe["close"] > dataframe["open"])
        )
        dataframe["sig_bb_short"] = (
            (dataframe["close"] >= dataframe["upper"]) & (dataframe["close"] < dataframe["open"])
        )

        dataframe["sig_macd_long"] = (
            (dataframe["macdhist"] > 0) & (dataframe["macdhist"] > dataframe["macdhist"].shift(1))
        )
        dataframe["sig_macd_short"] = (
            (dataframe["macdhist"] < 0) & (dataframe["macdhist"] < dataframe["macdhist"].shift(1))
        )

        dataframe["sig_rsi_long"] = (
            (dataframe["rsi"] > dataframe["rsi"].shift(1)) & (dataframe["rsi"] < 45)
        )
        dataframe["sig_rsi_short"] = (
            (dataframe["rsi"] < dataframe["rsi"].shift(1)) & (dataframe["rsi"] > 55)
        )

        # Composite confirmation scores (additive depth, not regression)
        dataframe["score_long"] = 0
        for s in [
            "sig_catchup_long", "sig_ema_long", "sig_bb_long",
            "sig_macd_long", "sig_rsi_long",
        ]:
            dataframe["score_long"] += dataframe[s].fillna(False).astype(int)

        dataframe["score_short"] = 0
        for s in [
            "sig_catchup_short", "sig_ema_short", "sig_bb_short",
            "sig_macd_short", "sig_rsi_short",
        ]:
            dataframe["score_short"] += dataframe[s].fillna(False).astype(int)

        dataframe["vol_ok"] = dataframe["volume"] > (dataframe["vol_ma"] * self.vol_mult)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        group, breadth, n = self._group_stats()
        if group is None or breadth is None:
            return dataframe
        # Require at least half the visible constellation to agree on direction
        if breadth < 0.5:
            return dataframe

        own = dataframe["ret_window"]
        gap = own - group

        if group >= self.group_move_min:
            # Constellation pumping: enter lagging members that show any soft confirmation
            is_laggard = gap <= -self.laggard_gap
            enter = is_laggard & (dataframe["score_long"] >= 1) & dataframe["vol_ok"]
            enter = enter & ~enter.shift(1).fillna(False)
            dataframe.loc[enter, "enter_long"] = 1
            dataframe.loc[enter, "enter_tag"] = f"BPL_g{group:+.1f}_brd{int(breadth*100)}"

        elif group <= -self.group_move_min:
            # Constellation dumping: short the lagging members
            is_laggard = gap >= self.laggard_gap
            enter = is_laggard & (dataframe["score_short"] >= 1) & dataframe["vol_ok"]
            enter = enter & ~enter.shift(1).fillna(False)
            dataframe.loc[enter, "enter_short"] = 1
            dataframe.loc[enter, "enter_tag"] = f"BPS_g{group:+.1f}_brd{int(breadth*100)}"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        group, breadth, n = self._group_stats()
        own = dataframe["ret_window"]
        gap = own - group if group is not None else None

        # Technical reversals (always active)
        rev_long = (
            qtpylib.crossed_below(dataframe["ema_f"], dataframe["ema_m"])
            | (dataframe["rsi"] > self.rsi_overbought)
        )
        rev_short = (
            qtpylib.crossed_above(dataframe["ema_f"], dataframe["ema_m"])
            | (dataframe["rsi"] < self.rsi_oversold)
        )

        exit_long = rev_long.fillna(False)
        exit_short = rev_short.fillna(False)

        if group is not None and gap is not None:
            # Caught up to the group
            exit_long = exit_long | (gap >= -self.catchup_exit)
            exit_short = exit_short | (gap <= self.catchup_exit)
            # Constellation cooled
            exit_long = exit_long | (group < self.group_exit_level)
            exit_short = exit_short | (group > -self.group_exit_level)

        exit_long = exit_long.fillna(False)
        exit_short = exit_short.fillna(False)

        dataframe.loc[exit_long, "exit_long"] = 1
        dataframe.loc[exit_long, "exit_tag"] = "bp_exit_long"
        dataframe.loc[exit_short, "exit_short"] = 1
        dataframe.loc[exit_short, "exit_tag"] = "bp_exit_short"

        return dataframe
