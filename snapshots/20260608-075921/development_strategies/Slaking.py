from pandas import DataFrame, merge
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
import talib.abstract as ta
import numpy as np

class Slaking(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '4h'
    stoploss = -0.08
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count = 150

    minimal_roi = {
        "0": 0.06,
        "2880": 0.03,
        "5760": 0.015
    }

    # Core mean-reversion parameters (preserved, narrowly tweaked per review)
    bb_period = IntParameter(20, 40, default=24, space="buy", optimize=False)
    bb_std = DecimalParameter(2.0, 3.5, default=2.2, space="buy", optimize=False)
    rsi_period = IntParameter(10, 20, default=14, space="buy", optimize=False)
    rsi_long_max = IntParameter(25, 45, default=38, space="buy", optimize=False)
    rsi_short_min = IntParameter(55, 75, default=62, space="buy", optimize=False)
    adx_period = IntParameter(10, 20, default=14, space="buy", optimize=False)
    adx_max = IntParameter(20, 40, default=28, space="buy", optimize=False)

    # Added depth: exhaustion confirmation & regime
    vol_ma_period = IntParameter(20, 40, default=24, space="buy", optimize=False)
    vol_mult = DecimalParameter(1.0, 2.5, default=1.3, space="buy", optimize=False)
    stoch_period = IntParameter(10, 20, default=14, space="buy", optimize=False)
    stoch_smooth = IntParameter(3, 5, default=3, space="buy", optimize=False)
    atr_period = IntParameter(10, 20, default=14, space="buy", optimize=False)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(p, '1d') for p in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Primary 4h indicators
        middle = dataframe['close'].rolling(window=self.bb_period.value).mean()
        std = dataframe['close'].rolling(window=self.bb_period.value).std()
        dataframe['bb_middleband'] = middle
        dataframe['bb_lowerband'] = middle - self.bb_std.value * std
        dataframe['bb_upperband'] = middle + self.bb_std.value * std
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        dataframe['volume_ma'] = dataframe['volume'].rolling(window=self.vol_ma_period.value).mean()
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        stoch = ta.STOCH(dataframe, fastk_period=self.stoch_period.value,
                         slowk_period=self.stoch_smooth.value, slowd_period=self.stoch_smooth.value)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        # Informative 1d for higher-timeframe regime
        if self.dp:
            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1d')
            if not informative.empty:
                informative['daily_rsi'] = ta.RSI(informative, timeperiod=14)
                informative['daily_adx'] = ta.ADX(informative, timeperiod=14)
                informative = informative[['date', 'daily_rsi', 'daily_adx']]
                dataframe = merge(dataframe, informative, on='date', how='left')
                dataframe.ffill(inplace=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0

        daily_rsi = dataframe.get('daily_rsi', 50)
        daily_adx = dataframe.get('daily_adx', 20)

        vol_ok = dataframe['volume'] > (dataframe['volume_ma'] * self.vol_mult.value)

        long_cond = (
            (dataframe['close'] < dataframe['bb_lowerband']) &
            (dataframe['rsi'] < self.rsi_long_max.value) &
            (dataframe['adx'] < self.adx_max.value) &
            (dataframe['stoch_k'] < 20) &
            (dataframe['stoch_k'] > dataframe['stoch_d']) &
            vol_ok &
            (daily_rsi < 70) &
            (daily_adx < 40)
        )
        dataframe.loc[long_cond, 'enter_long'] = 1

        short_cond = (
            (dataframe['close'] > dataframe['bb_upperband']) &
            (dataframe['rsi'] > self.rsi_short_min.value) &
            (dataframe['adx'] < self.adx_max.value) &
            (dataframe['stoch_k'] > 80) &
            (dataframe['stoch_k'] < dataframe['stoch_d']) &
            vol_ok &
            (daily_rsi > 30) &
            (daily_adx < 40)
        )
        dataframe.loc[short_cond, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

        long_exit = (
            (dataframe['close'] > dataframe['bb_middleband']) |
            (dataframe['rsi'] > 55)
        )
        dataframe.loc[long_exit, 'exit_long'] = 1

        short_exit = (
            (dataframe['close'] < dataframe['bb_middleband']) |
            (dataframe['rsi'] < 45)
        )
        dataframe.loc[short_exit, 'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: str | None, side: str, **kwargs) -> float:
        return 1.0

    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: str | None, side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) < 50:
            return proposed_stake
        last_atr = dataframe['atr'].iloc[-1]
        atr_mean = dataframe['atr'].rolling(window=50).mean().iloc[-1]
        if np.isfinite(last_atr) and np.isfinite(atr_mean) and last_atr > atr_mean * 1.5:
            return proposed_stake * 0.65
        return proposed_stake
