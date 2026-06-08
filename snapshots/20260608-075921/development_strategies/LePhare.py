from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class LePhare(IStrategy):
    """
    Le Phare: The Veteran Goalie (rev.1)
    Multi-timeframe EMA alignment (15m / 1h / 4h / 1d wind) with relaxed,
    composite confirmation filters. Selective but expressive—designed to
    survive ugly markets without sitting idle through valid trends.
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
    adx_threshold = 15
    rsi_long_max = 70
    rsi_short_min = 30
    rsi_exit_overbought = 80
    rsi_exit_oversold = 20

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
        # Three-horizon alignment (strict: 15m, 1h, 4h)
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

        # Relaxed confirmation components
        adx_ok = dataframe["adx"] > self.adx_threshold
        rsi_long_ok = dataframe["rsi"] < self.rsi_long_max
        rsi_short_ok = dataframe["rsi"] > self.rsi_short_min
        vol_ok = dataframe["volume"] > (dataframe["volume_sma"] * 0.7)
        atr_ok = dataframe["atr"] > (dataframe["atr_ma"] * 0.7)

        # Daily structural wind (optional depth)
        wind_long_1d = dataframe["ema_fast_1d"] > dataframe["ema_slow_1d"]
        wind_short_1d = dataframe["ema_fast_1d"] < dataframe["ema_slow_1d"]

        # Composite score: need 2 of 4 base filters, OR 1 of 4 if daily also aligns
        long_score = (
            adx_ok.astype(int) +
            rsi_long_ok.astype(int) +
            vol_ok.astype(int) +
            atr_ok.astype(int)
        )
        short_score = (
            adx_ok.astype(int) +
            rsi_short_ok.astype(int) +
            vol_ok.astype(int) +
            atr_ok.astype(int)
        )

        confirm_long = (long_score >= 2) | ((long_score >= 1) & wind_long_1d)
        confirm_short = (short_score >= 2) | ((short_score >= 1) & wind_short_1d)

        dataframe.loc[align_long & confirm_long, "enter_long"] = 1
        dataframe.loc[align_short & confirm_short, "enter_short"] = 1

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
