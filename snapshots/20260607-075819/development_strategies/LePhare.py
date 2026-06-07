from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class LePhare(IStrategy):
    """
    Le Phare: The Veteran Goalie.
    Multi-timeframe EMA alignment (15m base / 1h / 4h) with confirmation filters.
    Selective, chop-avoidant, designed to survive ugly markets.
    """

    # --- Core config -----------------------------------------------------
    timeframe = "15m"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 400

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
    rsi_long_max = 65
    rsi_short_min = 35
    rsi_exit_overbought = 75
    rsi_exit_oversold = 25

    # --- Futures leverage ------------------------------------------------
    def leverage(self, pair: str, current_time, rate: float,
                 proposed_leverage: float, **kwargs) -> float:
        """Conservative leverage for dry-run safety."""
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
        # Three-horizon alignment
        align_long = (
            (dataframe["ema_fast"] > dataframe["ema_slow"]) &
            (dataframe["ema_fast_1h"] > dataframe["ema_slow_1h"]) &
            (dataframe["ema_fast_4h"] > dataframe["ema_slow_4h"])
        )

        align_short = (
            (dataframe["ema_fast"] < dataframe["ema_slow"]) &
            (dataframe["ema_fast_1h"] < dataframe["ema_slow_1h"]) &
            (dataframe["ema_fast_4h"] < dataframe["ema_slow_4h"])
        )

        # Confirmation filters to avoid chop
        filters_long = (
            (dataframe["adx"] > self.adx_threshold) &
            (dataframe["rsi"] < self.rsi_long_max) &
            (dataframe["volume"] > dataframe["volume_sma"]) &
            (dataframe["atr"] > dataframe["atr_ma"])
        )

        filters_short = (
            (dataframe["adx"] > self.adx_threshold) &
            (dataframe["rsi"] > self.rsi_short_min) &
            (dataframe["volume"] > dataframe["volume_sma"]) &
            (dataframe["atr"] > dataframe["atr_ma"])
        )

        dataframe.loc[align_long & filters_long, "enter_long"] = 1
        dataframe.loc[align_short & filters_short, "enter_short"] = 1

        return dataframe

    # --- Exit logic ------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit on 15m reversal, 1h structural break, or RSI extreme
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
