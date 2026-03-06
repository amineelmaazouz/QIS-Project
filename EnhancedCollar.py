import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.stats import norm
import buywrite as buywrite

warnings.filterwarnings("ignore")


@dataclass
class Params:
    maturity_days: int
    k1_offset: float
    k2_offset: float
    kf: float
    forward_start_days: int
    transaction_cost: float = 0.001
    dividend_yield: float   = 0.0
    initial_level: float    = 100.0


class MarketDataEnhanced:

    def __init__(self, start_date: str = '2015-01-01', end_date: str = '2025-01-01'):
        self.start_date = start_date
        self.end_date   = end_date
        self.data       = None
        self.spx        = None
        self.vix        = None
        self.tbill      = None
        self._download_data()

    def _download_data(self):
        tickers  = ['^GSPC', '^VIX', '^IRX']
        print(f"Téléchargement des données de {self.start_date} à {self.end_date}")
        raw_data = yf.download(tickers, start=self.start_date,
                               end=self.end_date, auto_adjust=True)['Close']
        self.spx   = raw_data['^GSPC'].rename('SPX').dropna()
        self.vix   = raw_data['^VIX'].rename('VIX').dropna()
        self.tbill = (raw_data['^IRX'] / 100).rename('Rf').dropna()
        self.data  = pd.concat([self.spx, self.vix, self.tbill], axis=1).dropna()
        print(f"Données chargées : {len(self.data)} observations")

    def get_data(self) -> pd.DataFrame:
        return self.data

    def get_benchmark(self) -> pd.Series:
        return 100 * self.spx / self.spx.iloc[0]


class BlackScholesModel:

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float,
                   sigma: float, q: float = 0.0) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return np.exp(-q * T) * S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float,
                  sigma: float, q: float = 0.0) -> float:
        if T <= 0 or sigma <= 0:
            return max(K - S, 0.0)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - np.exp(-q * T) * S * norm.cdf(-d1)


class ForwardStartCallPricer:

    @staticmethod
    def price(S0: float, KF: float, T0: float, T: float,
              r: float, sigma: float, q: float = 0.0) -> float:
        residual = T - T0
        if residual <= 0 or sigma <= 0:
            return 0.0
        return S0 * BlackScholesModel.call_price(
            S=1.0, K=KF, T=residual, r=r, sigma=sigma, q=q
        )

    @staticmethod
    def payoff(ST0: float, ST: float, KF: float) -> float:
        return max(ST - KF * ST0, 0.0)


class EnhancedCollar:

    def __init__(self, market_data: MarketDataEnhanced, config: Params):
        self.market_data = market_data
        self.config      = config
        self.data        = market_data.get_data()
        self.bs          = BlackScholesModel()
        self.fsc_pricer  = ForwardStartCallPricer()
        self.il_history  = []
        self.roll_dates  = []
        self.trades      = []

    def _get_price_at(self, date: pd.Timestamp) -> float:
        idx   = self.data.index.get_loc(date)
        shift = self.config.forward_start_days
        if idx + shift < len(self.data):
            return self.data.iloc[idx + shift]['SPX']
        return self.data.iloc[-1]['SPX']

    def run_backtest(self) -> Tuple[pd.Series, buywrite.StrategyMetrics]:
        self.roll_dates = self.data.iloc[::self.config.maturity_days].index
        il = self.config.initial_level
        self.il_history = [il]
        for i in range(len(self.roll_dates) - 1):
            t0, t1 = self.roll_dates[i], self.roll_dates[i + 1]
            il *= (1 + self.perf(t0, t1))
            self.il_history.append(il)
        il_series = pd.Series(self.il_history, index=self.roll_dates)
        metrics   = self._calculate_metrics(il_series)
        return il_series, metrics

    def perf(self, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
        S0    = self.data.loc[t0, 'SPX']
        r     = self.data.loc[t0, 'Rf']
        sigma = self.data.loc[t0, 'VIX'] / 100
        T     = (t1 - t0).days / 252
        T0_yr = self.config.forward_start_days / 252

        K1 = S0 * (1 - self.config.k1_offset)
        K2 = S0 * (1 - self.config.k2_offset)

        put_long_prem   = self.bs.put_price(S0, K1, T, r, sigma, self.config.dividend_yield)
        put_short_prem  = self.bs.put_price(S0, K2, T, r, sigma, self.config.dividend_yield)
        put_spread_cost = put_long_prem - put_short_prem

        fsc_premium = self.fsc_pricer.price(
            S0=S0, KF=self.config.kf, T0=T0_yr, T=T,
            r=r, sigma=sigma, q=self.config.dividend_yield
        )

        ST  = self.data.loc[t1, 'SPX']
        ST0 = self._get_price_at(t0)

        stock_return      = (ST - S0) / S0
        put_long_payoff   = max(K1 - ST, 0.0)
        put_short_payoff  = max(K2 - ST, 0.0)
        put_spread_payoff = put_long_payoff - put_short_payoff
        put_spread_return = (put_spread_payoff - put_spread_cost) / S0
        fsc_payoff        = self.fsc_pricer.payoff(ST0, ST, self.config.kf)
        fsc_return        = (fsc_premium - fsc_payoff) / S0
        total_return      = stock_return + put_spread_return + fsc_return - self.config.transaction_cost

        self.trades.append({
            'date'             : t0,
            'spot'             : S0,
            'K1'               : K1,
            'K2'               : K2,
            'KF_abs'           : self.config.kf * ST0,
            'put_long_prem'    : put_long_prem,
            'put_short_prem'   : put_short_prem,
            'put_spread_cost'  : put_spread_cost,
            'fsc_premium'      : fsc_premium,
            'put_spread_payoff': put_spread_payoff,
            'fsc_payoff'       : fsc_payoff,
            'return'           : total_return,
        })

        return total_return

    def _calculate_metrics(self, il_series: pd.Series) -> buywrite.StrategyMetrics:
        returns       = il_series.pct_change().dropna()
        total_return  = il_series.iloc[-1] / il_series.iloc[0]
        n_periods     = len(returns)
        annual_return = total_return ** (252 / (self.config.maturity_days * n_periods)) - 1
        annual_vol    = returns.std() * np.sqrt(252 / self.config.maturity_days)
        sharpe        = annual_return / annual_vol if annual_vol > 0 else np.nan
        cummax        = il_series.cummax()
        max_dd        = ((cummax - il_series) / cummax).max()
        mat_label     = self._get_maturity_label()
        strike_label  = (f'K1-{int(self.config.k1_offset*100)}%/'
                         f'K2-{int(self.config.k2_offset*100)}%/'
                         f'KF+{int((self.config.kf-1)*100)}%')

        return buywrite.StrategyMetrics(
            maturity          = mat_label,
            strike            = strike_label,
            annual_return     = annual_return,
            annual_volatility = annual_vol,
            sharpe_ratio      = sharpe,
            max_drawdown      = max_dd,
            final_il          = il_series.iloc[-1],
        )

    def _get_maturity_label(self) -> str:
        return {10: '2W', 21: '1M', 63: '3M'}.get(
            self.config.maturity_days, f'{self.config.maturity_days}D'
        )

    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)


class EnhancedCollarOptimizer:

    def __init__(self, market_data: MarketDataEnhanced):
        self.market_data = market_data
        self.results     = {}
        self.all_metrics = []

    def run_optimization(self,
                         maturities: Dict[str, int],
                         k1_offsets: List[float],
                         k2_offsets: List[float],
                         kf_values: List[float],
                         forward_start_days: int = 21,
                         transaction_cost: float = 0.001) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("OPTIMISATION ENHANCED COLLAR")
        print("=" * 60)

        configs = [
            (mat_label, mat_days, k1, k2, kf)
            for mat_label, mat_days in maturities.items()
            for k1 in k1_offsets
            for k2 in k2_offsets
            for kf in kf_values
            if k2 > k1
        ]

        total = len(configs)
        for i, (mat_label, mat_days, k1, k2, kf) in enumerate(configs, 1):
            label = (f'{mat_label}|K1-{int(k1*100)}%/'
                     f'K2-{int(k2*100)}%/KF+{int((kf-1)*100)}%')
            print(f"[{i}/{total}] {label}", end="...")

            config = Params(
                maturity_days      = mat_days,
                k1_offset          = k1,
                k2_offset          = k2,
                kf                 = kf,
                forward_start_days = forward_start_days,
                transaction_cost   = transaction_cost,
            )

            strategy = EnhancedCollar(self.market_data, config)
            il_series, metrics = strategy.run_backtest()

            self.results[label] = il_series
            self.all_metrics.append(metrics)
            print(f" Sharpe={metrics.sharpe_ratio:.2f}")

        df_results = pd.DataFrame([
            {
                'Maturity'     : m.maturity,
                'Strike'       : m.strike,
                'Annual_Return': m.annual_return,
                'Annual_Vol'   : m.annual_volatility,
                'Sharpe'       : m.sharpe_ratio,
                'Max_DD'       : m.max_drawdown,
                'Final_IL'     : m.final_il,
            }
            for m in self.all_metrics
        ]).sort_values('Sharpe', ascending=False)

        return df_results

    def get_best_strategy(self) -> Tuple[str, buywrite.StrategyMetrics]:
        best  = max(self.all_metrics, key=lambda x: x.sharpe_ratio)
        label = f'{best.maturity}|{best.strike}'
        return label, best

    def plot_results(self, top_n: int = None):
        plt.figure(figsize=(14, 8))

        sorted_metrics  = sorted(self.all_metrics, key=lambda x: x.sharpe_ratio, reverse=True)
        metrics_to_plot = sorted_metrics[:top_n] if top_n else sorted_metrics

        for metrics in metrics_to_plot:
            label   = f'{metrics.maturity}|{metrics.strike}'
            series  = self.results[label]
            is_best = metrics == sorted_metrics[0]
            plt.plot(series.index, series,
                     label=label,
                     alpha=0.85 if is_best else 0.4,
                     linewidth=2.5 if is_best else 1)

        buy_hold = self.market_data.get_benchmark()
        plt.plot(buy_hold.index, buy_hold, 'k--', linewidth=2.5, label='Buy & Hold')

        plt.title("Enhanced Collar vs Buy & Hold (Base 100)", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Valeur cumulée (base 100)", fontsize=12)
        plt.legend(ncol=3, fontsize=8, loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def print_summary(self, top_n: int = 5):
        print("\n" + "=" * 60)
        print("RÉSULTATS ENHANCED COLLAR")
        print("=" * 60)

        best_label, best = self.get_best_strategy()
        print(f"\n Meilleure stratégie (Sharpe max) : {best_label}")
        print(f"  Rendement annuel   : {best.annual_return:.2%}")
        print(f"  Volatilité annuelle: {best.annual_volatility:.2%}")
        print(f"  Sharpe Ratio       : {best.sharpe_ratio:.2f}")
        print(f"  Max Drawdown       : {best.max_drawdown:.2%}")
        print(f"  IL final           : {best.final_il:.2f}")

        print(f"\n Top {top_n} stratégies :")
        top = sorted(self.all_metrics, key=lambda x: x.sharpe_ratio, reverse=True)[:top_n]
        for i, m in enumerate(top, 1):
            print(f"\n  {i}. {m.maturity}|{m.strike}")
            print(f"     Return: {m.annual_return:.2%} | Vol: {m.annual_volatility:.2%} | "
                  f"Sharpe: {m.sharpe_ratio:.2f} | MDD: {m.max_drawdown:.2%}")


def main():
    market_data = MarketDataEnhanced(start_date='2015-01-01', end_date='2025-01-01')

    maturities         = {'1M': 21, '3M': 63}
    k1_offsets         = [0.02, 0.05]
    k2_offsets         = [0.05, 0.10, 0.15]
    kf_values          = [1.00, 1.02, 1.05]
    forward_start_days = 21

    optimizer  = EnhancedCollarOptimizer(market_data)
    results_df = optimizer.run_optimization(
        maturities         = maturities,
        k1_offsets         = k1_offsets,
        k2_offsets         = k2_offsets,
        kf_values          = kf_values,
        forward_start_days = forward_start_days,
        transaction_cost   = 0.001,
    )

    optimizer.print_summary(top_n=5)
    results_df.to_csv('enhanced_collar_results.csv', index=False)
    print("\nRésultats sauvegardés dans 'enhanced_collar_results.csv'")
    optimizer.plot_results(top_n=8)


if __name__ == "__main__":
    main()