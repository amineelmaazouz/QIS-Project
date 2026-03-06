import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
warnings.filterwarnings("ignore")


@dataclass
class Params:
    maturity_days: int
    strike_offset: float  # en %
    transaction_cost: float = 0.001
    dividend_yield: float = 0.0

@dataclass()
class StrategyMetrics:
    maturity: str
    strike: str
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    final_il: float

    def __repr__(self):
        return (f"StrategyMetrics(mat={self.maturity}, strike={self.strike}, "
                f"ret={self.annual_return:.2%}, vol={self.annual_volatility:.2%}, "
                f"sharpe={self.sharpe_ratio:.2f}, mdd={self.max_drawdown:.2%})")


class BlackScholesModel:
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float,
                   sigma: float, q: float = 0.0) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call = np.exp(-q * T) * S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float,
                   sigma: float, q: float = 0.0) -> float:
        if T <= 0 or sigma <= 0:
            return max(K - S, 0.0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        put = K * np.exp(-r * T) * norm.cdf(-d2) - np.exp(-q * T) * S * norm.cdf(-d1)
        return put


class MarketData:

    def __init__(self, start_date: str = '2015-01-01', end_date: str = '2025-01-01'):
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.spx = None
        self.vix = None
        self.tbill = None
        self.bxm = None

        self._download_data()

    def _download_data(self):
        tickers = ['^GSPC', '^VIX', '^IRX', '^BXM']

        print(f"Téléchargement des données de {self.start_date} à {self.end_date}")
        raw_data = yf.download(tickers, start=self.start_date,
                               end=self.end_date, auto_adjust=True)['Close']

        self.spx = raw_data['^GSPC'].rename('SPX').dropna()
        self.vix = raw_data['^VIX'].rename('VIX').dropna()
        self.tbill = (raw_data['^IRX'] / 100).rename('Rf').dropna()  # Conversion en décimal
        self.bxm = raw_data['^BXM'].rename('BXM').dropna()
        self.data = pd.concat([self.spx, self.vix, self.tbill,self.bxm], axis=1).dropna()
        print(f"Données chargées : {len(self.data)} observations")

    def get_data(self) -> pd.DataFrame:
        return self.data

    def get_benchmark(self) -> pd.Series:
        return 100 * self.spx / self.spx.iloc[0]

    def get_bxm_benchmark(self) -> Optional[pd.Series]:
        if self.bxm is not None:
            return 100 * self.bxm / self.bxm.iloc[0]
        return None


class BuyWriteStrategy:
    """
    Stratégie Buy-Write (Covered Call) sur le S&P 500
    Long S&P 500 et short systématiquement des calls OTM
    """

    def __init__(self, market_data: MarketData, config: Params):

        self.market_data = market_data
        self.config = config
        self.data = market_data.get_data()
        self.pricing_model = BlackScholesModel()

        self.il_history = []
        self.roll_dates = []
        self.trades = []

    def run_backtest(self) :

        self.roll_dates = self.data.iloc[::self.config.maturity_days].index

        il = 100.0
        self.il_history = [il]

        for i in range(len(self.roll_dates) - 1):
            t0, t1 = self.roll_dates[i], self.roll_dates[i + 1]

            il *= (1 + self.perf(t0, t1))
            self.il_history.append(il)

        il_series = pd.Series(self.il_history, index=self.roll_dates)

        metrics = self._calculate_metrics(il_series)

        return il_series, metrics

    def perf(self, t0: pd.Timestamp, t1: pd.Timestamp) -> float:

        S0 = self.data.loc[t0, 'SPX']
        r = self.data.loc[t0, 'Rf']
        sigma = self.data.loc[t0, 'VIX'] / 100
        T = (t1 - t0).days / 252

        strike = S0 * (1 + self.config.strike_offset)

        call_premium = self.pricing_model.call_price(
            S=S0, K=strike, T=T, r=r, sigma=sigma, q=self.config.dividend_yield
        )


        ST = self.data.loc[t1, 'SPX']

        stock_return = (ST - S0) / S0

        call_payoff = max(ST - strike, 0)
        call_return = (call_premium - call_payoff) / S0

        total_return = stock_return + call_return - self.config.transaction_cost

        self.trades.append({
            'date': t0,
            'spot': S0,
            'strike': strike,
            'premium': call_premium,
            'payoff': call_payoff,
            'return': total_return
        })

        return total_return

    def _calculate_metrics(self, il_series: pd.Series) -> StrategyMetrics:

        returns = il_series.pct_change().dropna()
        total_return = il_series.iloc[-1] / il_series.iloc[0]
        n_periods = len(returns)
        annual_return = total_return ** (252 / (self.config.maturity_days * n_periods)) - 1
        annual_vol = returns.std() * np.sqrt(252 / self.config.maturity_days)
        sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
        cummax = il_series.cummax()
        drawdown = (cummax - il_series) / cummax
        max_dd = drawdown.max()
        mat_label = self._get_maturity_label()
        strike_label = f'+{int(self.config.strike_offset * 100)}%'

        return StrategyMetrics(
            maturity=mat_label,
            strike=strike_label,
            annual_return=annual_return,
            annual_volatility=annual_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            final_il=il_series.iloc[-1]
        )

    def _get_maturity_label(self) -> str:
        days_to_label = {10: '2W', 21: '1M', 63: '3M'}
        return days_to_label.get(self.config.maturity_days, f'{self.config.maturity_days}D')

    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)


class BuyWriteOptimizer:

    def __init__(self, market_data: MarketData):

        self.market_data = market_data
        self.results = {}
        self.all_metrics = []

    def run_optimization(self,
                         maturities: Dict[str, int],
                         strike_offsets: List[float],
                         transaction_cost: float = 0.001) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("OPTIMISATION BUY-WRITE")
        print("=" * 60)

        total_configs = len(maturities) * len(strike_offsets)
        current = 0

        for mat_label, mat_days in maturities.items():
            for strike_offset in strike_offsets:
                current += 1
                print(f"[{current}/{total_configs}] Test: {mat_label} | "
                      f"+{int(strike_offset * 100)}%", end="...")

                config = Params(
                    maturity_days=mat_days,
                    strike_offset=strike_offset,
                    transaction_cost=transaction_cost
                )

                strategy = BuyWriteStrategy(self.market_data, config)
                il_series, metrics = strategy.run_backtest()

                label = f'{mat_label}|+{int(strike_offset * 100)}%'
                self.results[label] = il_series
                self.all_metrics.append(metrics)

                print(f" Sharpe={metrics.sharpe_ratio:.2f}")

        df_results = pd.DataFrame([
            {
                'Maturity': m.maturity,
                'Strike': m.strike,
                'Annual_Return': m.annual_return,
                'Annual_Vol': m.annual_volatility,
                'Sharpe': m.sharpe_ratio,
                'Max_DD': m.max_drawdown,
                'Final_il': m.final_il
            }
            for m in self.all_metrics
        ]).sort_values('Sharpe', ascending=False)

        return df_results

    def get_best_strategy(self) -> Tuple[str, StrategyMetrics]:

        best_metrics = max(self.all_metrics, key=lambda x: x.sharpe_ratio)
        best_label = f'{best_metrics.maturity}|{best_metrics.strike}'
        return best_label, best_metrics

    def plot_results(self, top_n: int = None):

        plt.figure(figsize=(14, 8))

        sorted_metrics = sorted(self.all_metrics,
                                key=lambda x: x.sharpe_ratio,
                                reverse=True)

        metrics_to_plot = sorted_metrics[:top_n] if top_n else sorted_metrics

        for metrics in metrics_to_plot:
            label = f'{metrics.maturity}|{metrics.strike}'
            series = self.results[label]
            alpha = 0.8 if metrics == sorted_metrics[0] else 0.4
            linewidth = 2.5 if metrics == sorted_metrics[0] else 1
            plt.plot(series.index, series, label=label,
                     alpha=alpha, linewidth=linewidth)

        buy_hold = self.market_data.get_benchmark()
        plt.plot(buy_hold.index, buy_hold, 'k',
                 linewidth=2.5, label='Buy & Hold', linestyle='--')

        bxm = self.market_data.get_bxm_benchmark()
        if bxm is not None:
            plt.plot(bxm.index, bxm, 'g',
                     linewidth=2, label='BXM', linestyle='-.')

        plt.title("Stratégies Buy-Write vs Benchmarks (Base 100)",
                  fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Valeur cumulée (base 100)", fontsize=12)
        plt.legend(ncol=3, fontsize=9, loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def print_summary(self, top_n: int = 5):

        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)

        best_label, best_metrics = self.get_best_strategy()

        print("\n Meilleure stratégie (Sharpe max):")
        print(f"  Configuration : {best_label}")
        print(f"  Rendement annuel : {best_metrics.annual_return:.2%}")
        print(f"  Volatilité annuelle : {best_metrics.annual_volatility:.2%}")
        print(f"  Sharpe Ratio : {best_metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown : {best_metrics.max_drawdown:.2%}")
        print(f"  IL final : {best_metrics.final_il:.2f}")

        print(f"\n Top {top_n} strategies:")
        sorted_metrics = sorted(self.all_metrics,
                                key=lambda x: x.sharpe_ratio,
                                reverse=True)[:top_n]

        for i, m in enumerate(sorted_metrics, 1):
            print(f"\n  {i}. {m.maturity}|{m.strike}")
            print(f"     Return: {m.annual_return:.2%} | Vol: {m.annual_volatility:.2%} | "
                  f"Sharpe: {m.sharpe_ratio:.2f} | MDD: {m.max_drawdown:.2%}")



def main():
    market_data = MarketData(start_date='2015-01-01', end_date='2025-01-01')
    maturities = {
        '2W': 10,
        '1M': 21,
        '3M': 63
    }
    strike_offsets = [0.00, 0.01, 0.02, 0.05]

    optimizer = BuyWriteOptimizer(market_data)
    results_df = optimizer.run_optimization(
        maturities=maturities,
        strike_offsets=strike_offsets,
        transaction_cost=0.001
    )

    optimizer.print_summary(top_n=5)
    print("\nExport des résultats...")
    results_df.to_csv('buy_write_results.csv', index=False)
    print("Résultats sauvegardés dans 'buy_write_results.csv'")

    print("\n Génération des plots...")
    optimizer.plot_results(top_n=8)


if __name__ == "__main__":
    main()