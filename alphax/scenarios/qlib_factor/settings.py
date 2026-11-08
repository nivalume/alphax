"""Options of the qlib factor-mining scenario."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from alphax.core.promotion import MetricCriterion, MetricGuard, PromotionRule
from alphax.env.qlib import QlibEnvSettings

#: Twenty Alpha158 features used by RD-Agent as the starting feature set.
ALPHA20: dict[str, str] = {
    "RESI5": "Resi($close, 5)/$close",
    "WVMA5": "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)",
    "RSQR5": "Rsquare($close, 5)",
    "KLEN": "($high-$low)/$open",
    "RSQR10": "Rsquare($close, 10)",
    "CORR5": "Corr($close, Log($volume+1), 5)",
    "CORD5": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
    "CORR10": "Corr($close, Log($volume+1), 10)",
    "ROC60": "Ref($close, 60)/$close",
    "RESI10": "Resi($close, 10)/$close",
    "VSTD5": "Std($volume, 5)/($volume+1e-12)",
    "RSQR60": "Rsquare($close, 60)",
    "CORR60": "Corr($close, Log($volume+1), 60)",
    "WVMA60": "Std(Abs($close/Ref($close, 1)-1)*$volume, 60)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 60)+1e-12)",
    "STD5": "Std($close, 5)/$close",
    "RSQR20": "Rsquare($close, 20)",
    "CORD60": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)",
    "CORD10": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)",
    "CORR20": "Corr($close, Log($volume+1), 20)",
    "KLOW": "(Less($open, $close)-$low)/$open",
}


class SegmentSettings(BaseModel):
    train: tuple[str, str] = ("2008-01-01", "2014-12-31")
    valid: tuple[str, str] = ("2015-01-01", "2016-12-31")
    test: tuple[str, str] = ("2017-01-01", "2020-08-01")


class DebugSampleSettings(BaseModel):
    """The small sample the coder iterates on."""

    start: str = "2018-01-01"
    end: str = "2019-12-31"
    instruments: int = Field(default=100, ge=5)
    cut_fraction: float = Field(
        default=0.75, gt=0.1, lt=0.95, description="Where the look-ahead check truncates the sample"
    )


class BacktestSettings(BaseModel):
    topk: int = 50
    n_drop: int = 5
    account: float = 100_000_000
    deal_price: str = "close"
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_cost: float = 5.0
    limit_threshold: float | None = 0.095


class QlibFactorSettings(BaseModel):
    qlib: QlibEnvSettings = Field(default_factory=QlibEnvSettings)
    data_dir: Path = Field(
        default=Path("~/.alphax/data/qlib_factor"), description="Where the exported price-volume files are kept"
    )
    market: str = "csi300"
    benchmark: str = "SH000300"
    fields: list[str] = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
    data_start: str = "2008-01-01"
    data_end: str | None = None
    debug: DebugSampleSettings = Field(default_factory=DebugSampleSettings)
    segments: SegmentSettings = Field(default_factory=SegmentSettings)
    label: str = "Ref($close, -2)/Ref($close, -1) - 1"
    base_features: dict[str, str] = ALPHA20
    model: dict[str, Any] = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.2,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    }
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    ann_scaler: int = 252
    max_factors_per_experiment: int = Field(default=3, ge=1)
    factor_timeout: float = Field(default=900.0, description="Limit of one factor computation, seconds")
    max_nan_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    correlation_threshold: float = Field(
        default=0.99, gt=0.0, le=1.0, description="New factors this correlated with a kept factor are dropped"
    )
    promotion: PromotionRule = PromotionRule(
        primary=MetricCriterion(metric="IC"),
        guards=[
            MetricGuard(metric="1day.excess_return_with_cost.annualized_return", tolerance=0.02),
            MetricGuard(metric="1day.excess_return_with_cost.max_drawdown", tolerance=0.05),
        ],
    )
    use_critic: bool = Field(default=True, description="Add the LLM code review as the last evaluation stage")
    stage_weights: dict[str, float] = {"execution": 0.3, "format": 0.3, "look_ahead": 0.2, "critic": 0.2}

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    @property
    def end_date(self) -> str:
        return self.data_end or self.segments.test[1]
