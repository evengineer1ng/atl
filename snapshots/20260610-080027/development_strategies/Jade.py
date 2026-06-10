from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from datetime import timedelta
from typing import Optional

class Jade(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    stoploss = -0.01
    minimal_roi = {
        "0": 0.06,
        "60": 0.04,
        "120": 0.02,
        "180": 0.01
    }
    trailing_stop = True
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True
    can_short = True
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count = 200
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    basket_pairs = [
        "ETH/USDT:USDT", "BNB/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
        "ADA/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT", "DOT/USDT:USDT",
        "LINK/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT", "UNI/USDT:USDT",
        "ETC/USDT:USDT", "XLM/USDT:USDT", "TRX/USDT:USDT", "MATIC/USDT:USDT",
        "ARB/USDT:USDT", "OP/USDT:USDT", "NEAR/USDT:USDT", "INJ/USDT:USDT"
    ]

    def informative_pairs(self):
        pairs = [(p, self.timeframe) for p in self.basket_pairs]
        if "BTC/USDT:USDT" not in [x[0] for x in pairs]:
            pairs.append(("BTC/USDT:USDT", self.timeframe))
        return pairs

    def _merge_pair(self, dataframe: DataFrame, pair: str, prefix: str) -> DataFrame:
        df = self.dp.get_pair_dataframe(pair, self.timeframe)
        if df.empty or len(df) < self.startup_candle_count:
            return dataframe
        sub = df[['date', 'close', 'high', 'low', 'volume']].copy()
        sub[f'{prefix}_returns'] = sub['close'].pct_change()
        sub[f'{prefix}_rsi'] = ta.RSI(sub['close'], timeperiod=14)
        sub[f'{prefix}_atr'] = ta.ATR(sub['high'], sub['low'], sub['close'], timeperiod=14)
        sub[f'{prefix}_adx'] = ta.ADX(sub['high'], sub['low'], sub['close'], timeperiod=14)
        sub[f'{prefix}_vol'] = sub[f'{prefix}_returns'].rolling(window=20, min_periods=15).std()
        sub[f'{prefix}_sma20'] = ta.SMA(sub['close'], timeperiod=20)
        sub[f'{prefix}_above_sma20'] = (sub['close'] > sub[f'{prefix}_sma20']).astype(float)
        sub = sub[['date', f'{prefix}_returns', f'{prefix}_rsi', f'{prefix}_atr',
                   f'{prefix}_adx', f'{prefix}_vol', f'{prefix}_above_sma20']]
        dataframe = dataframe.merge(sub, on='date', how='left')
        for col in [c for c in dataframe.columns if c.startswith(prefix)]:
            dataframe[col] = dataframe[col].ffill(limit=3)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        dataframe['returns'] = dataframe['close'].pct_change()
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close']
        dataframe['adx'] = ta.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['volatility'] = dataframe['returns'].rolling(window=20, min_periods=15).std()
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20, min_periods=15).mean()
        dataframe['volume_z'] = (dataframe['volume'] - dataframe['volume_mean']) / dataframe['volume'].rolling(window=20, min_periods=15).std().replace(0, np.nan)
        dataframe['high_20'] = dataframe['high'].rolling(window=20, min_periods=15).max()
        dataframe['low_20'] = dataframe['low'].rolling(window=20, min_periods=15).min()
        dataframe['breakout_high'] = dataframe['close'] > dataframe['high_20'].shift(1)
        dataframe['breakout_low'] = dataframe['close'] < dataframe['low_20'].shift(1)
        dataframe['trending'] = dataframe['adx'] > 25
        dataframe['ranging'] = dataframe['adx'] < 20
        dataframe['strong_trend'] = dataframe['adx'] > 35
        dataframe['candle_bullish'] = dataframe['close'] > dataframe['open']
        dataframe['candle_bearish'] = dataframe['close'] < dataframe['open']

        if pair == "BTC/USDT:USDT":
            valid_prefixes = []
            for alt in self.basket_pairs:
                prefix = alt.replace('/', '_').replace(':', '_')
                dataframe = self._merge_pair(dataframe, alt, prefix)
                if f'{prefix}_returns' in dataframe.columns:
                    valid_prefixes.append(prefix)

            if valid_prefixes:
                ret_cols = [f'{p}_returns' for p in valid_prefixes]
                rsi_cols = [f'{p}_rsi' for p in valid_prefixes]
                vol_cols = [f'{p}_vol' for p in valid_prefixes]
                adx_cols = [f'{p}_adx' for p in valid_prefixes]
                breadth_cols = [f'{p}_above_sma20' for p in valid_prefixes]

                dataframe['basket_returns'] = dataframe[ret_cols].mean(axis=1)
                dataframe['basket_rsi'] = dataframe[rsi_cols].mean(axis=1)
                dataframe['basket_vol'] = dataframe[vol_cols].mean(axis=1)
                dataframe['basket_adx'] = dataframe[adx_cols].mean(axis=1)
                dataframe['basket_breadth'] = dataframe[breadth_cols].mean(axis=1)
                dataframe['dispersion'] = dataframe[ret_cols].std(axis=1)
            else:
                dataframe['basket_returns'] = 0.0
                dataframe['basket_rsi'] = 50.0
                dataframe['basket_vol'] = 1e-9
                dataframe['basket_adx'] = 20.0
                dataframe['basket_breadth'] = 0.5
                dataframe['dispersion'] = 0.0

            dataframe['divergence'] = dataframe['returns'] - dataframe['basket_returns']
            dataframe['divergence_rsi'] = dataframe['rsi'] - dataframe['basket_rsi']
            dataframe['vol_ratio'] = dataframe['volatility'] / (dataframe['basket_vol'] + 1e-9)
            dataframe['ref_adx'] = dataframe['basket_adx']

            dataframe['lead_lag_corr'] = dataframe['returns'].rolling(window=20, min_periods=15).corr(dataframe['basket_returns'])
            dataframe['btc_leads'] = dataframe['returns'].shift(1).rolling(window=20, min_periods=15).corr(dataframe['basket_returns'])
            dataframe['basket_leads'] = dataframe['basket_returns'].shift(1).rolling(window=20, min_periods=15).corr(dataframe['returns'])
            dataframe['lead_lag_score'] = dataframe['btc_leads'].fillna(0) - dataframe['basket_leads'].fillna(0)

            dataframe['research_group'] = 'btc_vs_basket'
        else:
            dataframe = self._merge_pair(dataframe, "BTC/USDT:USDT", "btc")
            dataframe['basket_returns'] = dataframe['btc_returns'].fillna(0)
            dataframe['basket_rsi'] = dataframe['btc_rsi'].fillna(50)
            dataframe['basket_vol'] = dataframe['btc_vol'].fillna(1e-9)
            dataframe['basket_adx'] = dataframe['btc_adx'].fillna(20)
            dataframe['basket_breadth'] = 0.5
            dataframe['dispersion'] = 0.0
            dataframe['divergence'] = dataframe['returns'] - dataframe['btc_returns'].fillna(0)
            dataframe['divergence_rsi'] = dataframe['rsi'] - dataframe['btc_rsi'].fillna(50)
            dataframe['vol_ratio'] = dataframe['volatility'] / (dataframe['btc_vol'].fillna(1e-9) + 1e-9)
            dataframe['ref_adx'] = dataframe['btc_adx'].fillna(20.0)
            dataframe['lead_lag_corr'] = 0.0
            dataframe['btc_leads'] = 0.0
            dataframe['basket_leads'] = 0.0
            dataframe['lead_lag_score'] = 0.0
            dataframe['research_group'] = 'alt_vs_btc'

        dataframe['divergence_ma'] = dataframe['divergence'].rolling(window=50, min_periods=30).mean()
        dataframe['divergence_std'] = dataframe['divergence'].rolling(window=50, min_periods=30).std()
        dataframe['divergence_z'] = (dataframe['divergence'] - dataframe['divergence_ma']) / (dataframe['divergence_std'] + 1e-9)

        vol_low = dataframe['volatility'].rolling(window=100, min_periods=50).quantile(0.25)
        vol_high = dataframe['volatility'].rolling(window=100, min_periods=50).quantile(0.75)
        dataframe['research_vol_regime'] = np.where(dataframe['volatility'] > vol_high, 'high_vol',
                                           np.where(dataframe['volatility'] < vol_low, 'low_vol', 'mid_vol'))

        dataframe['research_corr_regime'] = np.where(dataframe['lead_lag_corr'] > 0.7, 'high_corr',
                                            np.where(dataframe['lead_lag_corr'] < 0.3, 'low_corr', 'mid_corr'))

        dataframe['research_breadth_regime'] = np.where(dataframe['basket_breadth'] > 0.6, 'strong_breadth',
                                               np.where(dataframe['basket_breadth'] < 0.4, 'weak_breadth', 'neutral_breadth'))

        dataframe['research_lead_lag'] = np.where(dataframe['lead_lag_score'] > 0.1, 'btc_leads',
                                         np.where(dataframe['lead_lag_score'] < -0.1, 'basket_leads', 'neutral_lead'))

        dataframe['research_regime'] = np.where(dataframe['strong_trend'], 'strong_trend',
                                       np.where(dataframe['trending'], 'trending', 'ranging'))

        dataframe['research_divergence_state'] = np.where(dataframe['divergence_z'] > 2, 'extreme_premium',
                                                 np.where(dataframe['divergence_z'] < -2, 'extreme_discount', 'neutral'))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''

        volume_strong = dataframe['volume'] > (dataframe['volume_mean'] * 1.5)
        adx_gate = dataframe['adx'] > 28
        breakout_high_ok = dataframe['breakout_high'] & (dataframe['close'] > dataframe['high_20'].shift(1) + 0.25 * dataframe['atr'])
        breakout_low_ok = dataframe['breakout_low'] & (dataframe['close'] < dataframe['low_20'].shift(1) - 0.25 * dataframe['atr'])

        base_long = (
            breakout_high_ok & adx_gate & volume_strong &
            (dataframe['rsi'] < 75) & (dataframe['rsi'] > 35)
        )
        base_short = (
            breakout_low_ok & adx_gate & volume_strong &
            (dataframe['rsi'] > 25) & (dataframe['rsi'] < 65)
        )

        btc = metadata['pair'] == "BTC/USDT:USDT"

        h1 = base_long & (dataframe['divergence'] > 0)
        dataframe.loc[h1, 'enter_long'] = 1
        dataframe.loc[h1, 'enter_tag'] = 'h1_breakout_divergence_long'

        h2 = base_long & (dataframe['divergence_z'] < -2.0) & (dataframe['rsi'] > 40)
        dataframe.loc[h2 & (dataframe['enter_long'] == 0), 'enter_long'] = 1
        dataframe.loc[h2 & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'h2_divergence_revert_long'

        h3 = base_long & dataframe['ranging'].shift(1) & dataframe['trending'] & (dataframe['vol_ratio'] > 1.0)
        dataframe.loc[h3 & (dataframe['enter_long'] == 0), 'enter_long'] = 1
        dataframe.loc[h3 & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'h3_regime_ignition_long'

        h4 = base_short & (dataframe['divergence'] < 0)
        dataframe.loc[h4, 'enter_short'] = 1
        dataframe.loc[h4, 'enter_tag'] = 'h4_breakdown_divergence_short'

        h5 = base_short & (dataframe['divergence_z'] > 2.0) & (dataframe['rsi'] < 60)
        dataframe.loc[h5 & (dataframe['enter_short'] == 0), 'enter_short'] = 1
        dataframe.loc[h5 & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'h5_divergence_fade_short'

        h6 = base_short & (dataframe['vol_ratio'] > 1.5) & dataframe['strong_trend'] & (dataframe['rsi'] > 50)
        dataframe.loc[h6 & (dataframe['enter_short'] == 0), 'enter_short'] = 1
        dataframe.loc[h6 & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'h6_vol_climax_short'

        if btc:
            h7 = base_long & (dataframe['lead_lag_score'] > 0.1) & (dataframe['basket_breadth'] > 0.55)
            dataframe.loc[h7 & (dataframe['enter_long'] == 0), 'enter_long'] = 1
            dataframe.loc[h7 & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'h7_btc_leads_breadth_long'

            h8 = base_long & (dataframe['lead_lag_score'] < -0.1) & (dataframe['divergence_z'] < -0.5)
            dataframe.loc[h8 & (dataframe['enter_long'] == 0), 'enter_long'] = 1
            dataframe.loc[h8 & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'h8_basket_leads_catchup_long'

            h9 = base_long & (dataframe['lead_lag_corr'] < 0.3) & (dataframe['vol_ratio'] > 1.2)
            dataframe.loc[h9 & (dataframe['enter_long'] == 0), 'enter_long'] = 1
            dataframe.loc[h9 & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'h9_low_corr_solo_long'

            h10 = base_long & (dataframe['lead_lag_corr'] > 0.7) & (dataframe['dispersion'] < dataframe['dispersion'].rolling(50, min_periods=30).quantile(0.3))
            dataframe.loc[h10 & (dataframe['enter_long'] == 0), 'enter_long'] = 1
            dataframe.loc[h10 & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'h10_high_corr_ignition_long'

            h11 = base_short & (dataframe['basket_vol'] > dataframe['volatility'] * 1.3) & (dataframe['divergence'] < 0)
            dataframe.loc[h11 & (dataframe['enter_short'] == 0), 'enter_short'] = 1
            dataframe.loc[h11 & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'h11_alt_vol_favor_short'

            h12 = base_long & dataframe['strong_trend'] & (dataframe['vol_ratio'] > 1.2) & (dataframe['volatility'] > dataframe['volatility'].shift(5))
            dataframe.loc[h12 & (dataframe['enter_long'] == 0), 'enter_long'] = 1
            dataframe.loc[h12 & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'h12_btc_vol_expansion_long'

            h13 = base_long & (dataframe['basket_breadth'] > 0.6) & (dataframe['basket_breadth'] > dataframe['basket_breadth'].shift(3))
            dataframe.loc[h13 & (dataframe['enter_long'] == 0), 'enter_long'] = 1
            dataframe.loc[h13 & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'h13_breadth_expansion_long'

            h14 = base_short & (dataframe['basket_breadth'] < 0.4) & (dataframe['basket_breadth'] < dataframe['basket_breadth'].shift(3))
            dataframe.loc[h14 & (dataframe['enter_short'] == 0), 'enter_short'] = 1
            dataframe.loc[h14 & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'h14_breadth_contraction_short'
        else:
            hA_long = base_long & (dataframe['divergence_z'] < -1.0)
            dataframe.loc[hA_long & (dataframe['enter_long'] == 0), 'enter_long'] = 1
            dataframe.loc[hA_long & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'hA_alt_catchup_long'

            hA_short = base_short & (dataframe['divergence_z'] > 1.0)
            dataframe.loc[hA_short & (dataframe['enter_short'] == 0), 'enter_short'] = 1
            dataframe.loc[hA_short & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'hA_alt_premium_short'

        cooldown = 6
        prev_long = dataframe['enter_long'].rolling(window=cooldown, min_periods=1).sum().shift(1).fillna(0)
        prev_short = dataframe['enter_short'].rolling(window=cooldown, min_periods=1).sum().shift(1).fillna(0)

        dataframe.loc[prev_long > 0, 'enter_long'] = 0
        dataframe.loc[prev_long > 0, 'enter_tag'] = ''
        dataframe.loc[prev_short > 0, 'enter_short'] = 0
        dataframe.loc[prev_short > 0, 'enter_tag'] = ''

        dataframe['research_entry_gate'] = np.where(base_long | base_short, 'passed', 'blocked')
        dataframe['research_cooldown_active'] = np.where((prev_long > 0) | (prev_short > 0), 'yes', 'no')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''

        exit_long_rsi = (dataframe['rsi'] > 88) & ~dataframe['strong_trend']
        exit_short_rsi = (dataframe['rsi'] < 12) & ~dataframe['strong_trend']

        exit_long_div = (dataframe['divergence_z'] < -3.0) & dataframe['candle_bearish'] & (dataframe['adx'] < dataframe['adx'].shift(1))
        exit_short_div = (dataframe['divergence_z'] > 3.0) & dataframe['candle_bullish'] & (dataframe['adx'] < dataframe['adx'].shift(1))

        exit_regime_long = (dataframe['adx'] < 20) & (dataframe['adx'].shift(1) > 30) & dataframe['candle_bearish']
        exit_regime_short = (dataframe['adx'] < 20) & (dataframe['adx'].shift(1) > 30) & dataframe['candle_bullish']

        exit_corr_long = (dataframe['lead_lag_corr'] < 0.2) & (dataframe['lead_lag_corr'].shift(1) > 0.5)
        exit_corr_short = (dataframe['lead_lag_corr'] < 0.2) & (dataframe['lead_lag_corr'].shift(1) > 0.5)

        exit_breadth_long = dataframe['basket_breadth'] < 0.25
        exit_breadth_short = dataframe['basket_breadth'] > 0.75

        dataframe.loc[exit_long_rsi, 'exit_tag'] = 'r1_rsi_extreme_long'
        dataframe.loc[exit_long_div & ~exit_long_rsi, 'exit_tag'] = 'r3_divergence_revert_long'
        dataframe.loc[exit_short_rsi, 'exit_tag'] = 'r2_rsi_extreme_short'
        dataframe.loc[exit_short_div & ~exit_short_rsi, 'exit_tag'] = 'r4_divergence_revert_short'

        dataframe.loc[exit_regime_long & (dataframe['exit_long'] == 0), 'exit_long'] = 1
        dataframe.loc[exit_regime_long & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'r5_regime_shift_exit'

        dataframe.loc[exit_corr_long & (dataframe['exit_long'] == 0), 'exit_long'] = 1
        dataframe.loc[exit_corr_long & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'r6_corr_breakdown_long'

        dataframe.loc[exit_breadth_long & (dataframe['exit_long'] == 0), 'exit_long'] = 1
        dataframe.loc[exit_breadth_long & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'r7_breadth_collapse_long'

        dataframe.loc[exit_regime_short & (dataframe['exit_short'] == 0), 'exit_short'] = 1
        dataframe.loc[exit_regime_short & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'r5_regime_shift_exit'

        dataframe.loc[exit_corr_short & (dataframe['exit_short'] == 0), 'exit_short'] = 1
        dataframe.loc[exit_corr_short & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'r6_corr_breakdown_short'

        dataframe.loc[exit_breadth_short & (dataframe['exit_short'] == 0), 'exit_short'] = 1
        dataframe.loc[exit_breadth_short & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'r7_breadth_extreme_short'

        dataframe.loc[exit_long_rsi | exit_long_div, 'exit_long'] = 1
        dataframe.loc[exit_short_rsi | exit_short_div, 'exit_short'] = 1

        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs) -> Optional[str]:
        duration = current_time - trade.open_date_utc
        if current_profit < 0 and duration > timedelta(minutes=45):
            return 'time_stop_loss'
        if current_profit < 0.005 and duration > timedelta(minutes=90):
            return 'time_stop_flat'
        return None
