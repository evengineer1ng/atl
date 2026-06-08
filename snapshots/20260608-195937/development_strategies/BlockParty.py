from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class BlockParty(IStrategy):
    """
    Block Party: the cartographer (rev.1)

    Belief: markets organize themselves into temporary "constellations" of
    co-moving coins. Most traders stare at individual stars; Block Party maps the
    sky first. If a constellation is genuinely lighting up, at least one member
    usually hasn't fully expressed the move yet -- the group is right, the laggard
    is next. So it rides the quietest star that still clearly belongs (the
    Rudolph: smallest mover, most room left to shine).

    This is a discovery strategy, and the discovery is split across two layers:

      * THE SKY MAP (server-side, the league's custom pairlist). The constellation
        clustering runs in the resource-governance pipeline: it pulls ~150 coins,
        clusters them by 7d return correlation, finds the lit constellation(s),
        surfaces their laggards, and serves the result as the `block_party`
        RemotePairList universe. So this strategy's *whitelist is already the
        active constellation* -- it does not need to (and in Freqtrade cannot)
        trade outside it.

      * THE TIMING (this file). Within that served neighborhood it watches the
        cross-section: it measures the group's move (median recent return across
        the whitelist), confirms the constellation is lit, and enters the members
        that are lagging the group's direction AND just starting to turn toward
        it. It exits when the laggard catches up, the constellation cools, or the
        star reverses on its own.

    Entries/exits are tagged with the group move and how far behind the laggard
    was, for downstream telemetry.
    """

    # --- Core config -----------------------------------------------------
    timeframe = "15m"
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 160

    # Concentrated and patient: find the neighborhood, ride a couple of laggards.
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

    # --- Tunable parameters ----------------------------------------------
    lookback = 96               # bars for the "recent move" window (24h on 15m)
    min_group = 4               # need this many constellation members with data
    group_move_min = 2.0        # % median move for the constellation to count as "lit"
    laggard_gap = 3.0           # % a member must trail the group's direction to qualify
    group_exit_level = 0.5      # constellation considered "cooled" below this median move
    rsi_overbought = 78
    rsi_oversold = 22

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Cache the group snapshot per candle so we don't recompute the whole
        # cross-section once per pair within a single processing cycle.
        self._group_cache: dict = {}

    # --- Futures leverage ------------------------------------------------
    def leverage(self, pair: str, current_time, rate: float,
                 proposed_leverage: float, **kwargs) -> float:
        return 1.0

    # --- Cross-sectional group reference (the "constellation move") -------
    def _group_move(self) -> float | None:
        """Median recent return (%) across the current whitelist == how hard the
        constellation as a whole is moving right now. None if we can't see enough
        members yet (then the strategy simply doesn't open new trades)."""
        if not self.dp:
            return None
        try:
            pairs = self.dp.current_whitelist()
        except Exception:
            return None
        if not pairs:
            return None

        rets: list[float] = []
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
            if not past or not np.isfinite(past) or past <= 0:
                continue
            ret = (close.iloc[-1] / past - 1.0) * 100.0
            if np.isfinite(ret):
                rets.append(float(ret))
            if cache_key is None:
                cache_key = df["date"].iloc[-1]

        if len(rets) < self.min_group:
            return None
        median = float(np.median(rets))
        if cache_key is not None:
            self._group_cache[cache_key] = median
        return median

    # --- Per-pair indicators ---------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # This star's own recent move (same window as the group reference).
        dataframe["ret_window"] = dataframe["close"].pct_change(self.lookback) * 100.0

        dataframe["ema_f"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_m"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["vol_ma"] = dataframe["volume"].rolling(20).mean()

        # "Turning toward the move": a persistent early-uptrend state (short EMA
        # over long, momentum past the midline) rather than a one-bar crossover,
        # so the rising-edge entry below fires on the first bar the laggard flips
        # to joining -- and doesn't go dead once RSI climbs.
        dataframe["turn_up"] = (
            (dataframe["ema_f"] > dataframe["ema_m"]) & (dataframe["close"] > dataframe["ema_f"]) & (dataframe["rsi"] > 50)
        )
        dataframe["turn_down"] = (
            (dataframe["ema_f"] < dataframe["ema_m"]) & (dataframe["close"] < dataframe["ema_f"]) & (dataframe["rsi"] < 50)
        )
        # Soft liquidity floor so we don't try to join on a dead candle.
        dataframe["vol_ok"] = dataframe["volume"] > (dataframe["vol_ma"] * 0.5)
        return dataframe

    # --- Entry: ride the laggard of a lit constellation ------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        group = self._group_move()
        if group is None:
            return dataframe  # can't read the neighborhood -> sit out

        own = dataframe["ret_window"]

        if group >= self.group_move_min:
            # Constellation pumping: buy the members lagging the group's rise that
            # are starting to turn up.
            laggard_long = own <= (group - self.laggard_gap)
            enter = (laggard_long & dataframe["turn_up"] & dataframe["vol_ok"]).fillna(False)
            enter = enter & ~enter.shift(1).fillna(False)  # earliest trigger only
            dataframe.loc[enter, "enter_long"] = 1
            dataframe.loc[enter, "enter_tag"] = f"BPL g{group:+.1f}"

        elif group <= -self.group_move_min:
            # Constellation dumping: short the members lagging the group's fall.
            laggard_short = own >= (group + self.laggard_gap)
            enter = (laggard_short & dataframe["turn_down"] & dataframe["vol_ok"]).fillna(False)
            enter = enter & ~enter.shift(1).fillna(False)
            dataframe.loc[enter, "enter_short"] = 1
            dataframe.loc[enter, "enter_tag"] = f"BPS g{group:+.1f}"

        return dataframe

    # --- Exit: caught up, constellation cooled, or star reversed ---------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        group = self._group_move()
        own = dataframe["ret_window"]

        # Technical reversals always apply (so we can exit even if the group read
        # is unavailable).
        rev_long = qtpylib.crossed_below(dataframe["ema_f"], dataframe["ema_m"]) | (dataframe["rsi"] > self.rsi_overbought)
        rev_short = qtpylib.crossed_above(dataframe["ema_f"], dataframe["ema_m"]) | (dataframe["rsi"] < self.rsi_oversold)

        exit_long = rev_long.fillna(False)
        exit_short = rev_short.fillna(False)

        if group is not None:
            # Caught up to the group, or the constellation cooled off.
            exit_long = exit_long | (own >= group) | (group < self.group_exit_level)
            exit_short = exit_short | (own <= group) | (group > -self.group_exit_level)

        exit_long = exit_long.fillna(False)
        exit_short = exit_short.fillna(False)

        dataframe.loc[exit_long, "exit_long"] = 1
        dataframe.loc[exit_long, "exit_tag"] = "bp_long_exit"
        dataframe.loc[exit_short, "exit_short"] = 1
        dataframe.loc[exit_short, "exit_tag"] = "bp_short_exit"

        return dataframe
