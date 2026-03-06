import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import buywrite as buywrite

warnings.filterwarnings("ignore")


@dataclass
class Params:
    maturity_days: int
    call_strike_offset: float
    put_strike_offset: float
    transaction_cost: float = 0.001
    dividend_yield: float = 0.0
    initial_level: float = 100.0


class MarketDataCollar:

    def __init__(self, start_date: str = '2015-01-01', end_date: str = '2025-01-01'):
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.spx = None
        self.vix = None
        self.tbill = None
        self._download_data()

    def _download_data(self):
        tickers = ['^GSPC', '^VIX', '^IRX']
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


class Collar:

    def __init__(self, market_data: MarketDataCollar, config: Params):
        self.market_data   = market_data
        self.config        = config
        self.data          = market_data.get_data()
        self.pricing_model = buywrite.BlackScholesModel()
        self.il_history = []
        self.roll_dates = []
        self.trades     = []

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

        K_call = S0 * (1 + self.config.call_strike_offset)
        K_put  = S0 * (1 - self.config.put_strike_offset)

        call_premium = self.pricing_model.call_price(
            S=S0, K=K_call, T=T, r=r, sigma=sigma, q=self.config.dividend_yield
        )
        put_premium = self.pricing_model.put_price(
            S=S0, K=K_put, T=T, r=r, sigma=sigma, q=self.config.dividend_yield
        )

        ST           = self.data.loc[t1, 'SPX']
        stock_return = (ST - S0) / S0
        call_payoff  = max(ST - K_call, 0)
        call_return  = (call_premium - call_payoff) / S0
        put_payoff   = max(K_put - ST, 0)
        put_return   = (put_payoff - put_premium) / S0
        total_return = stock_return + call_return + put_return - self.config.transaction_cost

        self.trades.append({
            'date'       : t0,
            'spot'       : S0,
            'K_call'     : K_call,
            'K_put'      : K_put,
            'call_prem'  : call_premium,
            'put_prem'   : put_premium,
            'call_payoff': call_payoff,
            'put_payoff' : put_payoff,
            'return'     : total_return,
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
        strike_label  = f'C+{int(self.config.call_strike_offset * 100)}%/P-{int(self.config.put_strike_offset * 100)}%'

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


class CollarOptimizer:

    def __init__(self, market_data: MarketDataCollar):
        self.market_data = market_data
        self.results     = {}
        self.all_metrics = []

    def run_optimization(self,
                         maturities: Dict[str, int],
                         call_offsets: List[float],
                         put_offsets: List[float],
                         transaction_cost: float = 0.001) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("OPTIMISATION COLLAR")
        print("=" * 60)

        total_configs = len(maturities) * len(call_offsets) * len(put_offsets)
        current = 0

        for mat_label, mat_days in maturities.items():
            for c_off in call_offsets:
                for p_off in put_offsets:
                    current += 1
                    label = f'{mat_label}|C+{int(c_off*100)}%/P-{int(p_off*100)}%'
                    print(f"[{current}/{total_configs}] {label}", end="...")

                    config = Params(
                        maturity_days      = mat_days,
                        call_strike_offset = c_off,
                        put_strike_offset  = p_off,
                        transaction_cost   = transaction_cost,
                    )

                    strategy = Collar(self.market_data, config)
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

        plt.title("Stratégies Collar vs Buy & Hold (Base 100)", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Valeur cumulée (base 100)", fontsize=12)
        plt.legend(ncol=3, fontsize=8, loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def print_summary(self, top_n: int = 5):
        print("\n" + "=" * 60)
        print("RÉSULTATS")
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
    market_data  = MarketDataCollar(start_date='2015-01-01', end_date='2025-01-01')
    maturities   = {'2W': 10, '1M': 21, '3M': 63}
    call_offsets = [0.00, 0.01, 0.02, 0.05]
    put_offsets  = [0.00, 0.01, 0.02, 0.05]

    optimizer  = CollarOptimizer(market_data)
    results_df = optimizer.run_optimization(
        maturities       = maturities,
        call_offsets     = call_offsets,
        put_offsets      = put_offsets,
        transaction_cost = 0.001,
    )

    optimizer.print_summary(top_n=5)
    results_df.to_csv('collar_results.csv', index=False)
    print("\nRésultats sauvegardés dans 'collar_results.csv'")
    optimizer.plot_results(top_n=8)


if __name__ == "__main__":
    main()