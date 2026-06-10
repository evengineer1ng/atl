import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

import talib.abstract as ta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter

logger = logging.getLogger(__name__)


class SecondAct(IStrategy):
    """Event-lifecycle cartographer (research Phase 1, revision: tightened hard stop).

    Second Act trades the aftermath of extraordinary moves. It runs on the Big Movers
    pairlist (assets that JUST moved hard), locates the event on its own 15m frame, makes
    an explicit prediction of the "second act" (the enter_tag), and exits when the market
    reveals what actually happened (the exit_tag). The primary product is the
    Entry Tag -> Exit Tag confusion matrix, not PnL.

    Predictions (enter_tag, with direction):
      continuation   - the move is not finished; ride it.
      mean_reversion - the market overreacted; fade it back toward equilibrium.
      deadcat        - a short-lived counter-trend bounce before weakness resumes.
      base_building  - volatility will collapse and a new range will form.
      regime_change  - the event begins a durable new trend.

    Outcomes (exit_tag) are classified independently from the prediction in custom_exit
    (continuation_exit / mean_reversion_exit / deadcat_exit / base_building_exit /
    regime_change_exit), plus a bounded `second_act_timeout` so every trade resolves and
    gets a ground-truth label. All thresholds are Phase-1 and tunable via strategy_parameters.

    REVISION NOTE:
    - Hard stoploss tightened from -0.06 to -0.055 to keep executed losses inside the
      -6% risk budget after slippage.
    - custom_exit now carries an explicit -5% circuit-breaker (`second_act_stop`) that
      fires before the hard backstop is reached, preventing a repeat of the prior shift's
      -6.2% tail-risk breach.
    """

    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True          # outcomes are classified in custom_exit
    ignore_roi_if_entry_signal = False
    exit_profit_only = False

    # ema_slow(50) + atr-baseline(50) + event lookback need a healthy warmup.
    startup_candle_count = 120

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "emergency_exit": "market",
    }

    # ROI is DISABLED ({"0": 100} = effectively off). freqtrade's ROI engine closes a trade with a
    # bare, unclassifiable `roi` exit_reason and would also cap a runner — both fight the behaviour
    # model. With ROI off, custom_exit is the SOLE profit-taker, so every profitable exit carries a
    # classifiable signature tag (continuation/deadcat/mean_reversion/base_building/regime_change)
    # explaining what the market became. The stop is the only hard risk floor; the behaviour model
    # usually classifies a failing trade (reversal / clear-loser read) before the stop is reached.
    minimal_roi = {"0": 100}
    # REVISION: Tightened from -0.06 to -0.055. The prior shift saw a worst-trade close of -6.2%,
    # breaching the stated -6% budget due to execution slippage. This buffer keeps the realized
    # tail inside the -6% guardrail. A -5% circuit-breaker in custom_exit fires first.
    stoploss = -0.055

    # --- Event-detection / entry params (buy space) ---
    event_lookback = IntParameter(8, 32, default=16, space="buy")        # ~4h of 15m candles
    move_threshold = DecimalParameter(0.03, 0.15, default=0.06, decimals=3, space="buy")
    vol_threshold = DecimalParameter(1.2, 2.5, default=1.6, decimals=2, space="buy")
    rsi_overbought = IntParameter(65, 85, default=75, space="buy")
    rsi_oversold = IntParameter(15, 35, default=25, space="buy")
    trend_adx = IntParameter(20, 40, default=28, space="buy")

    # --- Outcome-classification / exit params (sell space) ---
    # Timeout sits BELOW the 6h dev shift (24 * 15m = 6h) so an unresolved trade still gets a
    # `second_act_timeout` label before the shift bell force-closes it (forced exits are
    # excluded from the research matrix). 20 candles = 5h.
    max_hold_candles = IntParameter(8, 24, default=20, space="sell")
    contraction_threshold = DecimalParameter(0.7, 1.3, default=1.05, decimals=2, space="sell")
    reversion_band = DecimalParameter(0.005, 0.05, default=0.02, decimals=3, space="sell")
    trend_separation = DecimalParameter(0.005, 0.05, default=0.015, decimals=3, space="sell")
    # Thesis-driven exit (behaviour model — NO fixed profit caps). A winner is granted GRACE to
    # run indefinitely while its entry thesis still looks high-quality: the trend still favours our
    # side (the entry signal hasn't deviated) AND profit is still growing (price near its best). It
    # banks — and classifies honestly by what the market became — only when the move reverses
    # against us or stalls, giving back more than a volatility-scaled trail from its peak. The move
    # itself decides how far it runs; there is no 8%/50% door.
    atr_trail_mult = DecimalParameter(0.8, 4.0, default=1.8, decimals=2, space="sell")   # trail = mult * atr_pct
    min_trail = DecimalParameter(0.006, 0.03, default=0.012, decimals=3, space="sell")   # floor so calm coins don't whipsaw
    profit_activation = DecimalParameter(0.004, 0.03, default=0.01, decimals=3, space="sell")  # arm the trail once a real winner

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return 1.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        lookback = int(self.event_lookback.value)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=50)

        # Volatility: ATR as a fraction of price, vs its own baseline (expansion ratio).
        atr = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = atr / dataframe["close"].replace(0, np.nan)
        atr_base = dataframe["atr_pct"].rolling(50).mean()
        dataframe["atr_ratio"] = dataframe["atr_pct"] / atr_base.replace(0, np.nan)

        # The event: signed move over the lookback window + where price sits in the
        # post-event range, plus a pre-event "equilibrium" proxy.
        ref = dataframe["close"].shift(lookback)
        dataframe["event_move"] = (dataframe["close"] - ref) / ref.replace(0, np.nan)
        recent_high = dataframe["high"].rolling(lookback).max()
        recent_low = dataframe["low"].rolling(lookback).min()
        span = (recent_high - recent_low).replace(0, np.nan)
        dataframe["pos_in_range"] = (dataframe["close"] - recent_low) / span
        dataframe["window_range"] = span / dataframe["close"].replace(0, np.nan)
        dataframe["sma_mid"] = ta.SMA(dataframe, timeperiod=lookback)

        dataframe = dataframe.fillna({"atr_ratio": 1.0, "pos_in_range": 0.5,
                                      "event_move": 0.0, "window_range": 0.0})
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        mv = float(self.move_threshold.value)
        ob, os_ = int(self.rsi_overbought.value), int(self.rsi_oversold.value)
        adx_t = int(self.trend_adx.value)
        contr = float(self.contraction_threshold.value)

        event_up = dataframe["event_move"] > mv
        event_down = dataframe["event_move"] < -mv
        had_event = dataframe["window_range"] > mv  # a big swing happened in the window

        # Priority cascade (most specific first); the enter_tag == "" guard makes the
        # first matching hypothesis win so each candle carries exactly one prediction.
        def assign(mask_long, mask_short, tag):
            free = dataframe["enter_tag"] == ""
            ml = mask_long & free & (dataframe["enter_short"] == 0)
            dataframe.loc[ml, ["enter_long", "enter_tag"]] = [1, f"{tag}_long"]
            free = dataframe["enter_tag"] == ""
            ms = mask_short & free & (dataframe["enter_long"] == 0)
            dataframe.loc[ms, ["enter_short", "enter_tag"]] = [1, f"{tag}_short"]

        # 1) regime_change: strong, aligned, sustained directional break.
        assign(
            event_up & (dataframe["adx"] > adx_t) & (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["close"] > dataframe["ema_slow"]) & (dataframe["pos_in_range"] > 0.6)
            & (dataframe["atr_ratio"] > 1.2),
            event_down & (dataframe["adx"] > adx_t) & (dataframe["ema_fast"] < dataframe["ema_slow"])
            & (dataframe["close"] < dataframe["ema_slow"]) & (dataframe["pos_in_range"] < 0.4)
            & (dataframe["atr_ratio"] > 1.2),
            "regime_change",
        )
        # 2) mean_reversion: overreaction at an extreme; fade it.
        assign(
            event_down & (dataframe["rsi"] < os_) & (dataframe["pos_in_range"] < 0.2),   # fade a crash -> long
            event_up & (dataframe["rsi"] > ob) & (dataframe["pos_in_range"] > 0.8),      # fade a pump -> short
            "mean_reversion",
        )
        # 3) deadcat: counter-trend bounce inside an intact opposing trend.
        assign(
            event_down & (dataframe["close"] > dataframe["open"]) & (dataframe["ema_fast"] < dataframe["ema_slow"])
            & (dataframe["rsi"] < 50) & (dataframe["pos_in_range"] < 0.5),
            event_up & (dataframe["close"] < dataframe["open"]) & (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["rsi"] > 50) & (dataframe["pos_in_range"] > 0.5),
            "deadcat",
        )
        # 4) continuation: momentum persists in the event direction (not yet exhausted).
        assign(
            event_up & (dataframe["ema_fast"] > dataframe["ema_slow"]) & (dataframe["rsi"] > 50)
            & (dataframe["rsi"] < ob) & (dataframe["pos_in_range"] > 0.55),
            event_down & (dataframe["ema_fast"] < dataframe["ema_slow"]) & (dataframe["rsi"] < 50)
            & (dataframe["rsi"] > os_) & (dataframe["pos_in_range"] < 0.45),
            "continuation",
        )
        # 5) base_building: post-event volatility collapse; trade the forming range edges.
        assign(
            had_event & (dataframe["atr_ratio"] < contr) & (dataframe["pos_in_range"] < 0.35),
            had_event & (dataframe["atr_ratio"] < contr) & (dataframe["pos_in_range"] > 0.65),
            "base_building",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    @staticmethod
    def _hypothesis(tag: str) -> str:
        """Reduce an entry tag (`<hyp>_long|short`) or exit tag (`<hyp>_exit`) to its
        underlying hypothesis so the entry prediction and a live outcome read can be compared."""
        t = str(tag or "").strip().lower()
        for suffix in ("_long", "_short", "_exit"):
            if t.endswith(suffix):
                return t[: -len(suffix)]
        return t

    def _detect_signature(self, c, trade) -> str:
        """Honest read of the second act the market is revealing RIGHT NOW (always one of the
        five outcome tags). Used both to label profit-taking exits truthfully and to drive the
        signature sniffer. Ordered most-specific first; the trend branches resolve everything
        else so a label is always produced."""
        close = float(c.get("close", 0.0) or 0.0)
        ema_fast = float(c.get("ema_fast", close) or close)
        ema_slow = float(c.get("ema_slow", close) or close)
        sma_mid = float(c.get("sma_mid", close) or close)
        atr_ratio = float(c.get("atr_ratio", 1.0) or 1.0)
        adx = float(c.get("adx", 0.0) or 0.0)
        window_range = float(c.get("window_range", 0.0) or 0.0)
        sep = abs(ema_fast - ema_slow) / close if close else 0.0

        if atr_ratio < float(self.contraction_threshold.value) and window_range < float(self.move_threshold.value):
            return "base_building_exit"               # volatility collapsed into a base
        if adx > int(self.trend_adx.value) and sep > float(self.trend_separation.value):
            return "regime_change_exit"               # strong, separated, durable trend
        if sma_mid and abs(close - sma_mid) / sma_mid < float(self.reversion_band.value):
            return "mean_reversion_exit"              # snapped back to equilibrium
        trend_up = ema_fast >= ema_slow
        if trend_up == (not trade.is_short):
            return "continuation_exit"                # market still travels the way we're positioned
        return "deadcat_exit"                         # market has turned against our side

    @staticmethod
    def _excursion(trade, current_rate: float) -> tuple[float, float]:
        """(peak, current) favourable excursion as gross return fractions in the trade's own
        direction, from freqtrade's tracked best price (max_rate for longs, min_rate for shorts).
        Lets the trail measure give-back from the BEST the move offered, scaling with whatever the
        move actually did — so 'profit growing' and 'profit stalled' are read off the move itself,
        with no fixed target."""
        open_rate = float(getattr(trade, "open_rate", 0.0) or 0.0)
        if open_rate <= 0:
            return 0.0, 0.0
        if trade.is_short:
            best = float(getattr(trade, "min_rate", None) or current_rate)
            return (open_rate - best) / open_rate, (open_rate - current_rate) / open_rate
        best = float(getattr(trade, "max_rate", None) or current_rate)
        return (best - open_rate) / open_rate, (current_rate - open_rate) / open_rate

    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[str]:
        """Thesis-quality behaviour model. Exits ONLY with honest outcome tags (the five `*_exit`
        reads or `second_act_timeout`). A winner is granted GRACE to run with NO profit ceiling
        while its entry thesis stays high-quality: the trend still favours our side (the entry
        signal hasn't deviated) AND profit is still near its peak (growing). It is banked —
        and classified by what the market became — only when the move turns against us or stalls,
        giving back more than a volatility-scaled trail from its best. Losers are still classified
        honestly; the hard stop and the timeout bound the rest. No fixed take-profit anywhere.

        REVISION: Added an explicit -5% circuit-breaker (`second_act_stop`) that exits via custom_exit
        BEFORE the -5.5% hard backstop is reached, eliminating the prior shift's -6.2% tail breach."""
        # Time-stop and hard stop are computed from the trade clock + live profit ONLY, so they
        # fire even when the analyzed dataframe is unavailable. PRIOR BUG: these sat BELOW an
        # `if dataframe.empty: return None` guard, so when the harness gave no dataframe (every
        # eval in practice) NOTHING fired and trades held to the shift bell / stop. The timeout
        # is the stale-thesis exit: a "second act" that has not developed in max_hold_candles is
        # sold, freeing the slot instead of riding it to the bell.
        duration_candles = int((current_time - trade.open_date_utc).total_seconds() // 60 // 15)
        if duration_candles >= int(self.max_hold_candles.value):
            return "second_act_timeout"
        if current_profit <= -0.05:
            return "second_act_stop"
        if duration_candles < 2:
            return None  # let the second act begin before judging it

        # Behaviour-model classification needs indicators — best-effort. If the dataframe is
        # unavailable we still have the time/stop bounds above, so a stale trade can no longer hide.
        if self.dp is None:
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None
        c = dataframe.iloc[-1]

        signature = self._detect_signature(c, trade)
        close = float(c.get("close", 0.0) or 0.0)
        ema_fast = float(c.get("ema_fast", close) or close)
        ema_slow = float(c.get("ema_slow", close) or close)
        sep = abs(ema_fast - ema_slow) / close if close else 0.0
        aligned = (ema_fast >= ema_slow) == (not trade.is_short)   # trend still favours our side

        peak, cur = self._excursion(trade, current_rate)
        atr_pct = float(c.get("atr_pct", 0.0) or 0.0)
        trail = max(float(self.atr_trail_mult.value) * atr_pct, float(self.min_trail.value))
        activated = peak >= float(self.profit_activation.value)     # has it been a real winner?

        # 1) Thesis turned against us: the trend no longer favours our side. The entry signal that
        #    carried the trade has deviated -> exit and classify what the market became (honest).
        if not aligned and (activated or current_profit < -0.005):
            return signature

        # 2) Move stalled: a real winner has given back more than a volatility-scaled trail from
        #    its peak (profit stopped growing). Momentum spent -> bank and classify the live read.
        if activated and (peak - cur) >= trail:
            return signature

        # 3) GRACE: thesis still high-quality (aligned) and profit still near its peak (growing).
        #    Hold with NO ceiling and let the move run as far as it will. (falls through)

        # 4) Honest classification for clear LOSING outcomes (never became a winner), so the matrix
        #    stays truthful on the downside too.
        if not activated and current_profit < 0:
            if signature in ("base_building_exit", "regime_change_exit", "mean_reversion_exit"):
                return signature
            if current_profit < -0.01 and sep > float(self.trend_separation.value):
                return "deadcat_exit"

        return None
