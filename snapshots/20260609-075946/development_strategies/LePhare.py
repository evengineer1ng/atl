from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class LePhare(IStrategy):
    """
    Le Phare: The Veteran Goalie (rev.2)
    Multi-timeframe EMA alignment (15m / 1h / 4h / 1d) with strict,
    high-quality entry qualification. Selective, confirmation-first
    trend following designed to survive ugly markets.
    """

    # --- Core config -----------------------------------------------------
    timeframe = "15m"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 2200

    # Risk / reward
    stoploss = -0.06
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03

    minimal_roi = {
        "0": 0.10,
        "120": 0.05,
        "240": 0.025,
    }

    # --- Tunable parameters ----------------------------------------------
    ema_fast = 8
    ema_slow = 21
    adx_threshold = 20
    rsi_long_min = 55
    rsi_long_max = 70
    rsi_short_max = 45
    rsi_short_min = 30
    rsi_exit_overbought = 80
    rsi_exit_oversold = 20
    vol_multiplier = 1.0
    atr_multiplier = 0.8
    proximity_atr_mult = 1.5

    # --- Futures leverage ------------------------------------------------
    def leverage(self, pair: str, current_time, rate: float,
                 proposed_leverage: float, **kwargs) -> float:
        return 1.0

    # --- Informative timeframes -----------------------------------------
    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow)
        return dataframe

    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow)
        return dataframe

    @informative("1d")
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow)
        return dataframe

    # --- Base timeframe indicators ---------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 15m trend
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow)

        # Confirmation / quality filters
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_ma"] = dataframe["atr"].rolling(window=14).mean()
        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()

        return dataframe

    # --- Entry logic -----------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 15m micro-structure: price leading and EMA rising
        price_above_fast = dataframe["close"] > dataframe["ema_fast"]
        fast_above_slow = dataframe["ema_fast"] > dataframe["ema_slow"]
        ema_fast_rising_15m = dataframe["ema_fast"] > dataframe["ema_fast"].shift(1)

        # 1h trend momentum (shift by 4 to compare prior 1h candle)
        ema_fast_above_slow_1h = dataframe["ema_fast_1h"] > dataframe["ema_slow_1h"]
        ema_fast_rising_1h = dataframe["ema_fast_1h"] > dataframe["ema_fast_1h"].shift(4)

        # 4h structural momentum (shift by 16 to compare prior 4h candle)
        ema_fast_above_slow_4h = dataframe["ema_fast_4h"] > dataframe["ema_slow_4h"]
        ema_fast_rising_4h = dataframe["ema_fast_4h"] > dataframe["ema_fast_4h"].shift(16)

        # 1d structural wind
        ema_fast_above_slow_1d = dataframe["ema_fast_1d"] > dataframe["ema_slow_1d"]

        # Strict multi-horizon alignment
        align_long = (
            price_above_fast &
            fast_above_slow &
            ema_fast_rising_15m &
            ema_fast_above_slow_1h &
            ema_fast_rising_1h &
            ema_fast_above_slow_4h &
            ema_fast_rising_4h &
            ema_fast_above_slow_1d
        )

        align_short = (
            (dataframe["close"] < dataframe["ema_fast"]) &
            (dataframe["ema_fast"] < dataframe["ema_slow"]) &
            (dataframe["ema_fast"] < dataframe["ema_fast"].shift(1)) &
            (dataframe["ema_fast_1h"] < dataframe["ema_slow_1h"]) &
            (dataframe["ema_fast_1h"] < dataframe["ema_fast_1h"].shift(4)) &
            (dataframe["ema_fast_4h"] < dataframe["ema_slow_4h"]) &
            (dataframe["ema_fast_4h"] < dataframe["ema_fast_4h"].shift(16)) &
            (dataframe["ema_fast_1d"] < dataframe["ema_slow_1d"])
        )

        # Trend strength
        adx_ok = dataframe["adx"] > self.adx_threshold

        # Momentum quality bands (avoid overextended / dead-bounce entries)
        rsi_long_ok = (dataframe["rsi"] > self.rsi_long_min) & (dataframe["rsi"] < self.rsi_long_max)
        rsi_short_ok = (dataframe["rsi"] < self.rsi_short_max) & (dataframe["rsi"] > self.rsi_short_min)

        # Volume confirmation
        vol_ok = dataframe["volume"] > (dataframe["volume_sma"] * self.vol_multiplier)

        # Volatility confirmation (avoid flat chop)
        atr_ok = dataframe["atr"] > (dataframe["atr_ma"] * self.atr_multiplier)

        # Candle confirmation (directional close)
        candle_long = dataframe["close"] > dataframe["open"]
        candle_short = dataframe["close"] < dataframe["open"]

        # Proximity guard: do not chase moves extended far from 15m fast EMA
        proximity_long = dataframe["close"] < (dataframe["ema_fast"] + self.proximity_atr_mult * dataframe["atr"])
        proximity_short = dataframe["close"] > (dataframe["ema_fast"] - self.proximity_atr_mult * dataframe["atr"])

        # All filters must pass simultaneously for a high-quality entry
        enter_long = (
            align_long &
            adx_ok &
            rsi_long_ok &
            vol_ok &
            atr_ok &
            candle_long &
            proximity_long
        )

        enter_short = (
            align_short &
            adx_ok &
            rsi_short_ok &
            vol_ok &
            atr_ok &
            candle_short &
            proximity_short
        )

        dataframe.loc[enter_long, "enter_long"] = 1
        dataframe.loc[enter_short, "enter_short"] = 1

        return dataframe

    # --- Exit logic ------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_exit = (
            qtpylib.crossed_below(dataframe["ema_fast"], dataframe["ema_slow"]) |
            qtpylib.crossed_below(dataframe["ema_fast_1h"], dataframe["ema_slow_1h"]) |
            (dataframe["rsi"] > self.rsi_exit_overbought)
        )

        short_exit = (
            qtpylib.crossed_above(dataframe["ema_fast"], dataframe["ema_slow"]) |
            qtpylib.crossed_above(dataframe["ema_fast_1h"], dataframe["ema_slow_1h"]) |
            (dataframe["rsi"] < self.rsi_exit_oversold)
        )

        dataframe.loc[long_exit, "exit_long"] = 1
        dataframe.loc[short_exit, "exit_short"] = 1

        return dataframe
