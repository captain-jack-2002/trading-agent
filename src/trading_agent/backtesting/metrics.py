"""Sample-period metrics. Undefined ratios are null, never fabricated infinity."""

from collections.abc import Sequence
from decimal import Decimal
from math import isfinite, sqrt
from statistics import mean, stdev


def compute_metrics(
    equity: Sequence[Decimal],
    trade_pnls: Sequence[Decimal],
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0,
    turnover: Decimal = Decimal(0),
    exposures: Sequence[Decimal] = (),
    costs: Decimal = Decimal(0),
    slippage: Decimal = Decimal(0),
) -> dict[str, float | int | None]:
    if not equity or any(not x.is_finite() or x < 0 for x in equity) or equity[0] <= 0:
        raise ValueError("positive initial and nonnegative finite subsequent equity required")
    if periods_per_year <= 0 or not isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("invalid annualization assumptions")
    pairs = list(zip(equity, equity[1:], strict=False))
    if any(a == 0 and b > 0 for a, b in pairs):
        raise ValueError("equity cannot recover from zero without external cash flows")
    returns = [float(b / a - 1) for a, b in pairs] if all(a > 0 for a, _ in pairs) else []
    total = float(equity[-1] / equity[0] - 1)
    annual = (
        (1 + total) ** (periods_per_year / len(returns)) - 1
        if len(returns) >= periods_per_year
        else None
    )
    vol = stdev(returns) * sqrt(periods_per_year) if len(returns) > 1 else None
    rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = [r - rf for r in returns]
    downside = sqrt(mean([min(r, 0) ** 2 for r in excess])) if excess else 0
    peak = equity[0]
    drawdown = Decimal(0)
    for value in equity:
        peak = max(peak, value)
        drawdown = max(drawdown, 1 - value / peak)
    wins = [float(p) for p in trade_pnls if p > 0]
    losses = [float(p) for p in trade_pnls if p < 0]
    return {
        "total_return": total,
        "annualized_return": annual,
        "annualized_volatility": vol,
        "sharpe": mean(excess) * periods_per_year / vol if vol and excess else None,
        "sortino": mean(excess) * sqrt(periods_per_year) / downside if downside else None,
        "max_drawdown": float(drawdown),
        "calmar": annual / float(drawdown) if drawdown and annual is not None else None,
        "win_rate": len(wins) / len(trade_pnls) if trade_pnls else None,
        "loss_rate": len(losses) / len(trade_pnls) if trade_pnls else None,
        "profit_factor": sum(wins) / -sum(losses) if losses else None,
        "average_win": mean(wins) if wins else None,
        "average_loss": mean(losses) if losses else None,
        "expectancy": mean([float(p) for p in trade_pnls]) if trade_pnls else None,
        "turnover": float(turnover / equity[0]),
        "closed_trade_count": len(trade_pnls),
        "average_exposure": mean([float(p) for p in exposures]) if exposures else 0,
        "costs": float(costs),
        "slippage": float(slippage),
    }


def compare_paper(
    simulated_equity: Sequence[Decimal],
    paper_equity: Sequence[Decimal],
    *,
    simulated_trade_count: int = 0,
    paper_trade_count: int = 0,
) -> dict[str, float | int]:
    """Caller aligns recorded observations; no database or account access."""
    if not simulated_equity or len(simulated_equity) != len(paper_equity):
        raise ValueError("nonempty aligned equity observations required")
    differences = [a - b for a, b in zip(simulated_equity, paper_equity, strict=True)]
    return {
        "final_equity_difference": float(differences[-1]),
        "mean_absolute_equity_difference": float(sum(map(abs, differences)) / len(differences)),
        "trade_count_difference": simulated_trade_count - paper_trade_count,
    }
