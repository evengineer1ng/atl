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
    """Event-lifecycle cartographer (research Phase 1).

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
    """

    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True
    process_only_new_candles = True
    use_exit_signal = False          # outcomes are classified in custom_exit
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

    # Loose ROI / protective stop so the lifecycle classifier (custom_exit) usually decides
    # the exit; ROI/stop are only safety nets. minimal_roi high => rarely preempts a label.
    minimal_roi = {"0": 0.12}
    stoploss = -0.06

    # --- Event-detection / entry params (buy space) ---
    event_lookback = IntParameter(8, 32, default=16, space="buy")        # ~4h of 15m candles
    move_threshold = DecimalParameter(0.03, 0.15, default=0.06, decimals=3, space="buy")
    vol_threshold = DecimalParameter(1.2, 2.5, default=1.6, decimals=2, space="buy")
    rsi_overbought = IntParameter(65, 85, default=75, space="buy")
    rsi_oversold = IntParameter(15, 35, default=25, space="buy")
    trend_adx = IntParameter(20, 40, default=28, space="buy")

    # --- Outcome-classification / exit params (sell space) ---
    max_hold_candles = IntParameter(12, 64, default=32, space="sell")    # ~8h ceiling
    contraction_threshold = DecimalParameter(0.7, 1.3, default=1.05, decimals=2, space="sell")
    reversion_band = DecimalParameter(0.005, 0.05, default=0.02, decimals=3, space="sell")
    trend_separation = DecimalParameter(0.005, 0.05, default=0.015, decimals=3, space="sell")

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

    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[str]:
        """Classify the OBSERVED outcome (ground truth) and exit with that tag. Independent
        of the entry prediction — the divergence is the research signal. Decision tree is
        evaluated in order; the first matching outcome wins. A max-hold timeout guarantees
        every trade resolves and receives a label."""
        if self.dp is None:
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None
        c = dataframe.iloc[-1]

        tf_minutes = 15
        duration_candles = int((current_time - trade.open_date_utc).total_seconds() // 60 // tf_minutes)
        if duration_candles < 2:
            return None  # let the second act begin before judging it

        if duration_candles >= int(self.max_hold_candles.value):
            return "second_act_timeout"

        atr_ratio = float(c.get("atr_ratio", 1.0) or 1.0)
        adx = float(c.get("adx", 0.0) or 0.0)
        close = float(c.get("close", 0.0) or 0.0)
        ema_fast = float(c.get("ema_fast", close) or close)
        ema_slow = float(c.get("ema_slow", close) or close)
        sma_mid = float(c.get("sma_mid", close) or close)
        window_range = float(c.get("window_range", 0.0) or 0.0)

        # base_building: volatility collapsed and the window range is tight (a base formed).
        if atr_ratio < float(self.contraction_threshold.value) and window_range < float(self.move_threshold.value):
            return "base_building_exit"

        # regime_change: trend is strong and the EMAs have separated decisively & aligned.
        sep = abs(ema_fast - ema_slow) / close if close else 0.0
        trend_dir = ema_fast >= ema_slow
        if adx > int(self.trend_adx.value) and sep > float(self.trend_separation.value) \
                and ((close > ema_slow) == trend_dir):
            return "regime_change_exit"

        # mean_reversion: price has returned to the pre-event equilibrium band.
        if sma_mid and abs(close - sma_mid) / sma_mid < float(self.reversion_band.value):
            return "mean_reversion_exit"

        # continuation: the move extended into profit and momentum is now rolling over.
        if current_profit > 0.01 and atr_ratio < 1.2:
            return "continuation_exit"

        # deadcat: the trade is underwater and the prior trend has reasserted (bounce faded).
        if current_profit < -0.01 and sep > float(self.trend_separation.value):
            return "deadcat_exit"

        return None
