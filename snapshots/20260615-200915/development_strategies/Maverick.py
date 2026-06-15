import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
from freqtrade.strategy import IStrategy
from pandas import DataFrame
from freqtrade.persistence import Trade

class Maverick(IStrategy):
    """
    Maverick: cross-sectional relative-outlier strategy for the Top 10 universe.
    Ranks all coins every 15m by a composite momentum score (15m, 1h, 4h).
    Enters long on the strongest positive outliers and short on the weakest
    negative outliers when divergence is accelerating.
    Exits are designed for asymmetric payoff: let winners run, cut losers fast.
    """
    INTERFACE_VERSION = 3

    can_short: bool = True
    timeframe = '15m'
    process_only_new_candles = True
    startup_candle_count = 20

    # --- EXIT GENOME PARAMETERS ---
    stoploss = -0.015
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.030

    minimal_roi = {
        '0': 0.035,
        '30': 0.025,
        '60': 0.015,
        '90': 0.005,
        '120': 0.0
    }

    use_exit_signal = False
    exit_profit_only = False

    # --- ENTRY GENOME PARAMETERS ---
    zscore_entry = 1.0
    min_dispersion = 0.003

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._cache_date: Optional[datetime] = None
        self._cache_stats: Optional[dict] = None

    # --- MANAGEMENT GENOME ---
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: Optional[str], side: str, **kwargs) -> float:
        return 2.0

    # --- ENTRY GENOME HELPERS ---
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

    # --- ENTRY GENOME ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['return_15m'] = dataframe['close'].pct_change(1)
        dataframe['return_1h'] = dataframe['close'].pct_change(4)
        dataframe['return_4h'] = dataframe['close'].pct_change(16)
        dataframe['composite'] = (
            0.5 * dataframe['return_15m']
            + 0.3 * dataframe['return_1h']
            + 0.2 * dataframe['return_4h']
        )
        dataframe['composite_prev'] = dataframe['composite'].shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        stats = self._universe_stats(dataframe, metadata)
        pair = metadata['pair']
        z = stats['zscores'].get(pair, 0.0)

        if pd.isna(z):
            return dataframe

        # Avoid trading when the universe is flat (noise)
        if stats['std'] < self.min_dispersion:
            return dataframe

        last_idx = dataframe.index[-1]
        current_composite = dataframe['composite'].iloc[-1]
        prev_composite = dataframe['composite_prev'].iloc[-1]

        if pd.isna(prev_composite) or pd.isna(current_composite):
            return dataframe

        # Approximate previous zscore using current universe stats (stable enough on 15m)
        z_prev = (prev_composite - stats['mean']) / stats['std'] if stats['std'] > 1e-9 else 0.0

        # Long: strong positive outlier that is accelerating
        if z > self.zscore_entry and z > z_prev and current_composite > 0:
            dataframe.loc[last_idx, 'enter_long'] = 1

        # Short: strong negative outlier that is accelerating
        elif z < -self.zscore_entry and z < z_prev and current_composite < 0:
            dataframe.loc[last_idx, 'enter_short'] = 1

        return dataframe

    # --- EXIT GENOME ---
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        # Time-stop: close losing trades after 90 minutes (6 candles) to recycle capital
        if trade.open_date is not None:
            duration = current_time - trade.open_date
            if duration >= timedelta(minutes=90) and current_profit < 0:
                return "time_stop"
        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Disabled: use_exit_signal=False. Previous zscore-based exit was bleeding.
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
