"""Pydantic models for type-safe data structures."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class WeatherData(BaseModel):
    """Weather data point."""

    timestamp: datetime
    temperature: float
    humidity: float = Field(ge=0, le=100)
    wind_speed: float = Field(ge=0)
    clouds: float = Field(ge=0, le=100)
    precipitation: float = Field(ge=0)
    pressure: Optional[float] = None
    location: Optional[str] = None
    description: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ForecastData(BaseModel):
    """Weather forecast data point."""

    timestamp: datetime
    temperature: float
    humidity: float = Field(ge=0, le=100)
    wind_speed: float = Field(ge=0)
    clouds: float = Field(ge=0, le=100)
    rain_probability: float = Field(ge=0, le=1)
    precipitation: float = Field(ge=0)
    description: Optional[str] = None
    source: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class MarketOpportunity(BaseModel):
    """Trading opportunity in a weather market."""

    market_id: str
    question: str
    location: str
    edge: float = Field(description="Predicted edge vs market price")
    liquidity_usd: float = Field(ge=0)
    volume_24h_usd: float = Field(ge=0)
    predicted_yes_probability: Optional[float] = Field(None, ge=0, le=1)
    current_yes_price: Optional[float] = Field(None, ge=0, le=1)
    predicted_no_probability: Optional[float] = Field(None, ge=0, le=1)
    current_no_price: Optional[float] = Field(None, ge=0, le=1)
    prediction_type: Optional[str] = None
    position_size: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class Trade(BaseModel):
    """Trading recommendation/executed trade."""

    market_id: str
    question: str
    location: str
    action: str = Field(description="BUY_YES or BUY_NO")
    position_size: float = Field(gt=0)
    entry_price: float = Field(ge=0, le=1)
    implied_probability: float = Field(ge=0, le=1)
    edge: float
    profit_target: float = Field(description="Profit target as percentage")
    stop_loss: float = Field(description="Stop loss as percentage")
    max_hold_time: int = Field(description="Max hold time in seconds")
    entry_time: datetime
    status: str = Field(default="PENDING")

    model_config = ConfigDict(from_attributes=True)


class Position(BaseModel):
    """Open trading position."""

    market_id: str
    action: str
    position_size: float
    entry_price: float
    entry_time: datetime
    current_price: Optional[float] = None
    current_pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    status: str = Field(default="OPEN")

    model_config = ConfigDict(from_attributes=True)


class PortfolioStats(BaseModel):
    """Portfolio performance statistics."""

    initial_capital: float
    current_capital: float
    total_pnl: float
    total_pnl_percent: float
    total_trades: int
    closed_trades: int
    daily_loss: float
    circuit_breaker_triggered: bool
    avg_pnl: float
    win_rate: float = Field(ge=0, le=1)

    model_config = ConfigDict(from_attributes=True)


class RiskConfig(BaseModel):
    """Risk management configuration."""

    initial_capital: float = Field(default=10000, gt=0)
    max_daily_loss_percent: float = Field(default=2.0, gt=0)
    max_concurrent_positions: int = Field(default=5, gt=0)
    min_position_size: float = Field(default=10, gt=0)
    max_position_size: float = Field(default=1000, gt=0)
    max_concentration: float = Field(default=0.30, ge=0, le=1)
    max_correlation: float = Field(default=0.70, ge=0, le=1)

    model_config = ConfigDict(from_attributes=True)


class TradingConfig(BaseModel):
    """Trading configuration."""

    edge_threshold: float = Field(default=0.05, gt=0)
    min_position_size: float = Field(default=10, gt=0)
    max_position_size: float = Field(default=1000, gt=0)
    max_concurrent_positions: int = Field(default=5, gt=0)
    profit_target: float = Field(default=0.20)
    stop_loss: float = Field(default=-0.10)
    time_based_exit: int = Field(default=86400, description="Max hold time in seconds")
    position_sizing_method: str = Field(default="kelly")

    model_config = ConfigDict(from_attributes=True)


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    enabled: bool = Field(default=True)
    daily_loss_threshold: float = Field(default=0.05, ge=0, le=1)
    max_consecutive_losses: int = Field(default=5, gt=0)

    model_config = ConfigDict(from_attributes=True)


class ModelMetrics(BaseModel):
    """Model training metrics."""

    train_accuracy: float = Field(ge=0, le=1)
    train_auc: float = Field(ge=0, le=1)
    train_precision: float = Field(ge=0, le=1)
    train_recall: float = Field(ge=0, le=1)
    train_f1: float = Field(ge=0, le=1)
    val_accuracy: Optional[float] = Field(None, ge=0, le=1)
    val_auc: Optional[float] = Field(None, ge=0, le=1)
    val_precision: Optional[float] = Field(None, ge=0, le=1)
    val_recall: Optional[float] = Field(None, ge=0, le=1)
    val_f1: Optional[float] = Field(None, ge=0, le=1)
    cv_mean: Optional[float] = Field(None, ge=0, le=1)
    cv_std: Optional[float] = Field(None, ge=0)

    model_config = ConfigDict(from_attributes=True)


class FeatureImportance(BaseModel):
    """Feature importance scores."""

    feature_name: str
    importance_score: float = Field(ge=0)
    rank: int = Field(gt=0)

    model_config = ConfigDict(from_attributes=True)


class CrossValidationResult(BaseModel):
    """Cross-validation results."""

    cv_mean: float = Field(ge=0, le=1)
    cv_std: float = Field(ge=0)
    cv_scores: List[float]

    model_config = ConfigDict(from_attributes=True)


class TradeLog(BaseModel):
    """Log entry for a trade or position close."""

    timestamp: datetime
    trade_id: Optional[str] = None
    market_id: str
    action: str
    position_size: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str

    model_config = ConfigDict(from_attributes=True)


class APIResponse(BaseModel):
    """Standard API response structure."""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


class PredictionResult(BaseModel):
    """Model prediction result."""

    market_id: str
    predicted_probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    prediction_timestamp: datetime
    model_version: str

    model_config = ConfigDict(from_attributes=True)
