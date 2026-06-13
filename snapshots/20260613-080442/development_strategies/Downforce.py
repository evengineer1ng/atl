from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame, Series
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Downforce(IStrategy):
    """
    Downforce: full-lifecycle waveform recognition (rev.1)

    Thesis (from brief): some moves carry a recognizable "downforce" signature -
    a waveform that recurs across several market phases before a high-conviction
    directional move. The edge is NOT detecting compression, breakout, or trend
    in isolation; it is recognizing the *full-pattern resemblance* across the
    lifecycle and then taking the earliest 1m trigger once the shape qualifies.

    The lifecycle is modelled as seven "wind-tunnel" stages, each scored 0..1 as
    a resemblance to the expected Downforce shape for that stage:
        1. compression
        2. early expansion
        3. breakout
        4. trend confirmation
        5. trend maturity
        6. exhaustion          (exit / against-trade)
        7. reversal            (exit / against-trade)

    Stages are treated as lenses, not gates: earlier stages must have been seen
    recently *in order* (rolling-max lookbacks), the late qualifying stages must
    be fresh, and a weighted composite similarity score must clear a high bar.
    Execution is on 1m so the entry fires immediately on the qualifying candle,
    but the pattern itself is built from longer 5m/15m/1h context. The strategy
    is expected to be quiet for long stretches and decisive when it acts.

    Entries/exits are tagged with the firing stage and the composite similarity
    score (e.g. "DFL:73") for downstream telemetry.
    """

    # --- Core config -----------------------------------------------------
    timeframe = "1m"
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    # 1m base needs deep history for the 1h context + long rolling baselines.
    startup_candle_count = 600

    # Risk / reward. High-conviction, concentrated; let the move breathe but
    # cut hard on invalidation (signal exits do most of the structural work).
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    minimal_roi = {
        "0": 0.05,
        "30": 0.03,
        "90": 0.015,
        "240": 0.0,
    }

    # --- Tunable parameters ----------------------------------------------
    # Stage resemblance floors (each 0..1) that must be cleared simultaneously.
    comp_floor = 0.40          # compression must have been seen recently
    expand_floor = 0.35        # followed by an early-expansion thrust
    breakout_floor = 0.45      # fresh breakout shape on the trigger candle
    confirm_floor = 0.45       # trend confirmation aligned with the breakout
    # Composite similarity required to call it a Downforce fingerprint.
    entry_threshold = 0.62
    # Exit sensitivity.
    exhaustion_exit = 0.62

    _EPS = 1e-9

    # --- Futures leverage ------------------------------------------------
    def leverage(self, pair: str, current_time, rate: float,
                 proposed_leverage: float, **kwargs) -> float:
        return 1.0

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _c01(series: Series, lo: float, hi: float) -> Series:
        """Clip a series onto 0..1 by a linear lo->hi map."""
        return ((series - lo) / (hi - lo)).clip(lower=0.0, upper=1.0)

    # --- Higher-timeframe context (the "longer lookback") ----------------
    @informative("5m")
    def populate_indicators_5m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=21)
        return dataframe

    @informative("15m")
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=21)
        return dataframe

    # --- Base (1m) indicators + stage scoring ----------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        eps = self._EPS

        # Trend skeleton
        dataframe["ema_f"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_m"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_s"] = ta.EMA(dataframe, timeperiod=55)

        # Directional / strength
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["roc"] = ta.ROC(dataframe, timeperiod=10)

        # Volatility envelope
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_upper"] = bb["upper"]
        dataframe["bb_lower"] = bb["lower"]
        dataframe["bb_mid"] = bb["mid"]
        dataframe["bbw"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / (dataframe["bb_mid"] + eps)

        # Range structure & participation
        dataframe["hh"] = dataframe["high"].rolling(20).max()
        dataframe["ll"] = dataframe["low"].rolling(20).min()
        dataframe["vol_ma"] = dataframe["volume"].rolling(30).mean()

        atr = dataframe["atr"]
        bbw = dataframe["bbw"]
        close = dataframe["close"]

        # ---- Stage 1: compression ---------------------------------------
        # Volatility coiled vs its own recent baseline (both band-width and ATR).
        bbw_ratio = bbw / (bbw.rolling(120).median() + eps)
        atr_ratio = atr / (atr.rolling(120).median() + eps)
        dataframe["s_compression"] = (
            self._c01(1.3 - bbw_ratio, 0.0, 0.8) * self._c01(1.3 - atr_ratio, 0.0, 0.8)
        )

        # ---- Stage 2: early expansion -----------------------------------
        # Volatility turning up off the coil.
        atr_mom = atr / (atr.shift(5) + eps) - 1.0
        dataframe["s_expansion"] = self._c01(atr_mom, 0.02, 0.30)

        # ---- Stage 3: breakout (directional) ----------------------------
        prior_hh = dataframe["hh"].shift(1)
        prior_ll = dataframe["ll"].shift(1)
        vol_ratio = dataframe["volume"] / (dataframe["vol_ma"] + eps)
        vol_score = self._c01(vol_ratio, 1.0, 2.5)
        dataframe["s_breakout_long"] = (
            self._c01((close - prior_hh) / (atr + eps), 0.0, 1.0) * vol_score
        )
        dataframe["s_breakout_short"] = (
            self._c01((prior_ll - close) / (atr + eps), 0.0, 1.0) * vol_score
        )

        # ---- Stage 4: trend confirmation (directional) ------------------
        adx_rising = dataframe["adx"] - dataframe["adx"].shift(3)
        ema_stack_long = ((dataframe["ema_f"] > dataframe["ema_m"]) & (dataframe["ema_m"] > dataframe["ema_s"])).astype(float)
        ema_stack_short = ((dataframe["ema_f"] < dataframe["ema_m"]) & (dataframe["ema_m"] < dataframe["ema_s"])).astype(float)
        di_spread = dataframe["plus_di"] - dataframe["minus_di"]
        adx_quality = self._c01(dataframe["adx"], 18.0, 35.0)
        rising_score = self._c01(adx_rising, 0.0, 10.0)
        dataframe["s_confirm_long"] = (
            0.45 * ema_stack_long
            + 0.30 * self._c01(di_spread, 0.0, 25.0)
            + 0.15 * adx_quality
            + 0.10 * rising_score
        )
        dataframe["s_confirm_short"] = (
            0.45 * ema_stack_short
            + 0.30 * self._c01(-di_spread, 0.0, 25.0)
            + 0.15 * adx_quality
            + 0.10 * rising_score
        )

        # ---- Stage 5: trend maturity (directional) ----------------------
        mature = self._c01(dataframe["adx"], 22.0, 40.0)
        dataframe["s_maturity_long"] = mature * ema_stack_long
        dataframe["s_maturity_short"] = mature * ema_stack_short

        # ---- Stage 6: exhaustion (against trade) ------------------------
        ext = (close - dataframe["ema_s"]) / (atr + eps)
        dataframe["s_exhaustion_long"] = (
            0.6 * self._c01(dataframe["rsi"], 72.0, 88.0) + 0.4 * self._c01(ext, 3.0, 9.0)
        )
        dataframe["s_exhaustion_short"] = (
            0.6 * self._c01(28.0 - dataframe["rsi"], 0.0, 16.0) + 0.4 * self._c01(-ext, 3.0, 9.0)
        )

        # ---- Composite Downforce similarity (ordered lifecycle) ---------
        # Earlier stages must have appeared recently and *before* the trigger:
        # compression sometime in the last ~90 bars (>=3 bars ago), then an
        # expansion thrust in the last ~45 bars. The breakout/confirmation are
        # evaluated fresh on the current candle below.
        comp_seen = dataframe["s_compression"].rolling(90).max().shift(3)
        expand_seen = dataframe["s_expansion"].rolling(45).max().shift(1)
        dataframe["comp_seen"] = comp_seen
        dataframe["expand_seen"] = expand_seen

        dataframe["df_long"] = (
            0.18 * comp_seen
            + 0.14 * expand_seen
            + 0.26 * dataframe["s_breakout_long"]
            + 0.26 * dataframe["s_confirm_long"]
            + 0.16 * dataframe["s_maturity_long"]
        ).fillna(0.0)
        dataframe["df_short"] = (
            0.18 * comp_seen
            + 0.14 * expand_seen
            + 0.26 * dataframe["s_breakout_short"]
            + 0.26 * dataframe["s_confirm_short"]
            + 0.16 * dataframe["s_maturity_short"]
        ).fillna(0.0)

        return dataframe

    # --- Entry logic -----------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Higher-timeframe directional bias: the fingerprint only counts when the
        # broader context agrees with the breakout direction.
        htf_long = (
            (dataframe["ema_fast_15m"] > dataframe["ema_slow_15m"])
            & (dataframe["ema_fast_1h"] > dataframe["ema_slow_1h"])
        )
        htf_short = (
            (dataframe["ema_fast_15m"] < dataframe["ema_slow_15m"])
            & (dataframe["ema_fast_1h"] < dataframe["ema_slow_1h"])
        )

        qualify_long = (
            (dataframe["comp_seen"] >= self.comp_floor)
            & (dataframe["expand_seen"] >= self.expand_floor)
            & (dataframe["s_breakout_long"] >= self.breakout_floor)
            & (dataframe["s_confirm_long"] >= self.confirm_floor)
            & htf_long
            & (dataframe["df_long"] >= self.entry_threshold)
        ).fillna(False)

        qualify_short = (
            (dataframe["comp_seen"] >= self.comp_floor)
            & (dataframe["expand_seen"] >= self.expand_floor)
            & (dataframe["s_breakout_short"] >= self.breakout_floor)
            & (dataframe["s_confirm_short"] >= self.confirm_floor)
            & htf_short
            & (dataframe["df_short"] >= self.entry_threshold)
        ).fillna(False)

        # Earliest trigger after full qualification: take the rising edge only,
        # so we enter on the first candle the shape completes and not on every
        # subsequent candle it stays qualified.
        enter_long = qualify_long & ~qualify_long.shift(1).fillna(False)
        enter_short = qualify_short & ~qualify_short.shift(1).fillna(False)

        tag_long = "DFL:" + (dataframe["df_long"] * 100).round().astype("Int64").astype(str)
        tag_short = "DFS:" + (dataframe["df_short"] * 100).round().astype("Int64").astype(str)

        dataframe.loc[enter_long, "enter_long"] = 1
        dataframe.loc[enter_long, "enter_tag"] = tag_long
        dataframe.loc[enter_short, "enter_short"] = 1
        dataframe.loc[enter_short, "enter_tag"] = tag_short

        return dataframe

    # --- Exit logic ------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Reversal stage: structural turn against the trade.
        rev_long = qtpylib.crossed_below(dataframe["ema_f"], dataframe["ema_m"]) | (
            dataframe["minus_di"] > dataframe["plus_di"]
        )
        rev_short = qtpylib.crossed_above(dataframe["ema_f"], dataframe["ema_m"]) | (
            dataframe["plus_di"] > dataframe["minus_di"]
        )

        # Invalidation: trend strength collapsing out of a mature move.
        adx_collapse = ((dataframe["adx"] - dataframe["adx"].shift(3)) < -8.0) & (dataframe["adx"] < 20.0)

        # Higher-timeframe context flips against the position.
        htf_flip_long = dataframe["ema_fast_1h"] < dataframe["ema_slow_1h"]
        htf_flip_short = dataframe["ema_fast_1h"] > dataframe["ema_slow_1h"]

        exit_long = (
            (dataframe["s_exhaustion_long"] >= self.exhaustion_exit)
            | rev_long
            | adx_collapse
            | htf_flip_long
        ).fillna(False)
        exit_short = (
            (dataframe["s_exhaustion_short"] >= self.exhaustion_exit)
            | rev_short
            | adx_collapse
            | htf_flip_short
        ).fillna(False)

        dataframe.loc[exit_long, "exit_long"] = 1
        dataframe.loc[exit_long, "exit_tag"] = "df_long_exit"
        dataframe.loc[exit_short, "exit_short"] = 1
        dataframe.loc[exit_short, "exit_tag"] = "df_short_exit"

        return dataframe
