import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class Maverick(IStrategy):
    '''
    Maverick: cross-sectional relative-outlier strategy for the Top 10 universe.
    Ranks all coins every 15m by a composite momentum score (15m, 1h, 4h).
    Enters long on the strongest positive outliers and short on the weakest
    negative outliers. Exits quickly when the outlier signal fades.
    '''
    INTERFACE_VERSION = 3

    can_short: bool = True
    timeframe = '15m'
    stoploss = -0.03
    trailing_stop = False

    minimal_roi = {
        '0': 0.02,
        '15': 0.01,
        '30': 0.005,
        '60': 0.0
    }

    process_only_new_candles = True
    startup_candle_count = 20
    use_exit_signal = True
    exit_profit_only = False

    zscore_entry = 1.0
    zscore_exit = 0.5

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._cache_date: Optional[datetime] = None
        self._cache_stats: Optional[dict] = None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: Optional[str], side: str, **kwargs) -> float:
        return 2.0

    def _composite_score(self, df: DataFrame) -> float:
        if df is None or len(df) < 16:
            return np.nan
        close = df['close']
        r15 = close.pct_change(1).iloc[-1]
        r1h = close.pct_change(4).iloc[-1]
        r4h = close.pct_change(16).iloc[-1]
        if pd.isna(r15) or pd.isna(r1h) or pd.isna(r4h):
            return np.nan
        return 0.5 * r15 + 0.3 * r1h + 0.2 * r4h

    def _universe_stats(self, dataframe: DataFrame, metadata: dict) -> dict:
        if self.dp is None or not hasattr(self.dp, 'current_whitelist'):
            return {'mean': 0.0, 'std': 1e-9, 'zscores': {}}

        current_date = dataframe['date'].iloc[-1]

        if self._cache_date == current_date and self._cache_stats is not None:
            return self._cache_stats

        whitelist = self.dp.current_whitelist()
        scores = {}

        for pair in whitelist:
            if pair == metadata['pair']:
                df = dataframe
            else:
                df = self.dp.get_pair_dataframe(pair, self.timeframe)

            if df is None or df.empty:
                continue

            df = df[df['date'] <= current_date]
            if len(df) < 16:
                continue

            scores[pair] = self._composite_score(df)

        if len(scores) < 3:
            return {'mean': 0.0, 'std': 1e-9, 'zscores': scores}

        series = pd.Series(scores)
        mean = series.mean()
        std = series.std()
        if pd.isna(std) or std == 0:
            std = 1e-9

        zscores = ((series - mean) / std).to_dict()
        stats = {'mean': mean, 'std': std, 'zscores': zscores}
        self._cache_date = current_date
        self._cache_stats = stats
        return stats

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['return_15m'] = dataframe['close'].pct_change(1)
        dataframe['return_1h'] = dataframe['close'].pct_change(4)
        dataframe['return_4h'] = dataframe['close'].pct_change(16)
        dataframe['composite'] = (
            0.5 * dataframe['return_15m']
            + 0.3 * dataframe['return_1h']
            + 0.2 * dataframe['return_4h']
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        stats = self._universe_stats(dataframe, metadata)
        pair = metadata['pair']
        z = stats['zscores'].get(pair, 0.0)

        if pd.isna(z):
            return dataframe

        last_idx = dataframe.index[-1]

        if z > self.zscore_entry:
            dataframe.loc[last_idx, 'enter_long'] = 1
        elif z < -self.zscore_entry:
            dataframe.loc[last_idx, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        stats = self._universe_stats(dataframe, metadata)
        pair = metadata['pair']
        z = stats['zscores'].get(pair, 0.0)

        if pd.isna(z):
            return dataframe

        last_idx = dataframe.index[-1]

        if z < self.zscore_exit:
            dataframe.loc[last_idx, 'exit_long'] = 1

        if z > -self.zscore_exit:
            dataframe.loc[last_idx, 'exit_short'] = 1

        return dataframe
