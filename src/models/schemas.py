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


# Configuration Models

class WeatherSource(BaseModel):
    """Weather data source configuration."""

    name: str
    enabled: bool = True
    api_key: str
    base_url: Optional[str] = None
    free_tier: bool = False
    endpoints: List[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class Location(BaseModel):
    """Geographic location configuration."""

    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    country: Optional[str] = None
    timezone: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class WeatherConfig(BaseModel):
    """Weather data collection configuration."""

    sources: List[WeatherSource] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    refresh_interval: int = Field(default=3600, gt=0)
    historical_lookback: int = Field(default=30, gt=0)

    model_config = ConfigDict(from_attributes=True)


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    lookback_windows: List[str] = Field(default_factory=lambda: ['3h', '6h', '12h', '24h'])
    aggregations: List[str] = Field(default_factory=list)
    seasonal_features: bool = True
    include_nan_imputation: bool = True
    imputation_method: str = "forward_fill"

    model_config = ConfigDict(from_attributes=True)


class XGBoostParams(BaseModel):
    """XGBoost model hyperparameters."""

    max_depth: int = Field(default=6, ge=1)
    learning_rate: float = Field(default=0.05, gt=0)
    n_estimators: int = Field(default=300, gt=0)
    subsample: float = Field(default=0.8, ge=0, le=1)
    colsample_bytree: float = Field(default=0.8, ge=0, le=1)
    objective: str = "binary:logistic"
    eval_metric: str = "auc"
    random_state: int = 42
    verbosity: int = Field(default=0, ge=0)

    model_config = ConfigDict(from_attributes=True)


class ModelConfig(BaseModel):
    """Machine learning model configuration."""

    algorithm: str = "xgboost"
    target_variable: Optional[str] = None
    xgboost_params: XGBoostParams = Field(default_factory=XGBoostParams)
    train_test_split: float = Field(default=0.8, ge=0, le=1)
    validation_split: float = Field(default=0.1, ge=0, le=1)
    cross_validation_folds: int = Field(default=5, ge=2)
    retrain_interval: int = Field(default=604800, gt=0)
    min_samples_for_retrain: int = Field(default=1000, gt=0)

    model_config = ConfigDict(from_attributes=True)


class MarketMappingConfig(BaseModel):
    """Market mapping configuration."""

    data_source: Optional[str] = None
    auto_discovery: bool = True
    discovery_interval: int = Field(default=86400, gt=0)
    market_filters: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_threshold: float = Field(default=0.65, ge=0, le=1)
    min_market_liquidity: float = Field(default=100, ge=0)

    model_config = ConfigDict(from_attributes=True)


class PositionSizingConfig(BaseModel):
    """Position sizing configuration."""

    method: str = Field(default="kelly", pattern="^(kelly|fixed)$")
    kelly_fraction: float = Field(default=0.25, ge=0, le=1)
    fixed_size_percent: float = Field(default=2.0, gt=0)

    model_config = ConfigDict(from_attributes=True)


class EntryCondition(BaseModel):
    """Entry condition configuration."""

    edge_requirement: Optional[bool] = None
    min_liquidity: Optional[float] = Field(None, ge=0)
    max_slippage: Optional[float] = Field(None, ge=0, le=1)

    model_config = ConfigDict(from_attributes=True)


class ExitCondition(BaseModel):
    """Exit condition configuration."""

    profit_target: Optional[float] = None
    stop_loss: Optional[float] = None
    time_based: Optional[int] = Field(None, gt=0)

    model_config = ConfigDict(from_attributes=True)


class TradingStrategyConfig(BaseModel):
    """Trading strategy configuration."""

    strategy: str = "probability_edge"
    edge_threshold: float = Field(default=0.05, gt=0)
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    entry_conditions: List[EntryCondition] = Field(default_factory=list)
    exit_conditions: List[ExitCondition] = Field(default_factory=list)
    max_concurrent_positions: int = Field(default=5, ge=1)
    min_position_size: float = Field(default=10, gt=0)
    max_position_size: float = Field(default=1000, gt=0)

    model_config = ConfigDict(from_attributes=True)


class PortfolioConfig(BaseModel):
    """Portfolio configuration."""

    initial_capital: float = Field(default=10000, gt=0)
    max_daily_loss_percent: float = Field(default=2.0, gt=0)
    max_drawdown_percent: float = Field(default=10.0, gt=0)
    max_leverage: float = Field(default=1.0, gt=0)

    model_config = ConfigDict(from_attributes=True)


class PositionConfig(BaseModel):
    """Position configuration."""

    max_risk_per_trade: float = Field(default=0.02, ge=0, le=1)
    max_concentration: float = Field(default=0.30, ge=0, le=1)
    max_correlation: float = Field(default=0.70, ge=0, le=1)

    model_config = ConfigDict(from_attributes=True)


class CircuitBreakerTriggers(BaseModel):
    """Circuit breaker triggers configuration."""

    api_failures: Optional[int] = Field(None, ge=1)
    risk_limit_breached: Optional[bool] = None
    connectivity_loss: Optional[int] = Field(None, gt=0)

    model_config = ConfigDict(from_attributes=True)


class EmergencyShutdownConfig(BaseModel):
    """Emergency shutdown configuration."""

    enabled: bool = True
    triggers: CircuitBreakerTriggers = Field(default_factory=CircuitBreakerTriggers)

    model_config = ConfigDict(from_attributes=True)


class RiskManagementConfig(BaseModel):
    """Risk management configuration."""

    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    position: PositionConfig = Field(default_factory=PositionConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    emergency_shutdown: EmergencyShutdownConfig = Field(default_factory=EmergencyShutdownConfig)

    model_config = ConfigDict(from_attributes=True)


class LearningConfig(BaseModel):
    """Learning loop configuration."""

    enabled: bool = True
    feedback: Dict[str, Any] = Field(default_factory=dict)
    retraining: Dict[str, Any] = Field(default_factory=dict)
    hyperparameter_tuning: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class PolymarketConfig(BaseModel):
    """Polymarket API configuration."""

    api_url: str = "https://api.polymarket.com"
    ws_url: Optional[str] = None
    network: Dict[str, Any] = Field(default_factory=dict)
    order_book_update_interval: int = Field(default=5, gt=0)
    price_update_interval: int = Field(default=1, gt=0)
    order_execution: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: Optional[str] = None
    file: Optional[str] = None
    max_file_size: int = Field(default=10485760, gt=0)
    backup_count: int = Field(default=5, ge=0)
    console_output: bool = True

    model_config = ConfigDict(from_attributes=True)


class DashboardConfig(BaseModel):
    """Dashboard configuration."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = Field(default=5000, ge=1, le=65535)
    debug: bool = False
    update_interval: int = Field(default=5, gt=0)
    features: List[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class NotificationChannel(BaseModel):
    """Notification channel configuration."""

    type: str
    enabled: bool = True
    path: Optional[str] = None
    smtp_server: Optional[str] = None
    smtp_port: Optional[int] = Field(None, ge=1, le=65535)
    from_address: Optional[str] = None
    webhook_url: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class NotificationsConfig(BaseModel):
    """Notifications configuration."""

    enabled: bool = True
    channels: List[NotificationChannel] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class MonitoringConfig(BaseModel):
    """Performance monitoring configuration."""

    enabled: bool = True
    metrics: List[str] = Field(default_factory=list)
    export_interval: int = Field(default=3600, gt=0)
    export_format: str = "json"

    model_config = ConfigDict(from_attributes=True)


class BacktestConfig(BaseModel):
    """Backtest configuration."""

    enabled: bool = False
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = Field(default=10000, gt=0)
    commission: float = Field(default=0.001, ge=0, le=1)
    slippage: float = Field(default=0.005, ge=0, le=1)

    model_config = ConfigDict(from_attributes=True)


class AgentConfig(BaseModel):
    """Complete agent configuration."""

    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    market_mapping: MarketMappingConfig = Field(default_factory=MarketMappingConfig)
    trading: TradingStrategyConfig = Field(default_factory=TradingStrategyConfig)
    risk_management: RiskManagementConfig = Field(default_factory=RiskManagementConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    polymarket: PolymarketConfig = Field(default_factory=PolymarketConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create config from dictionary with validation."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump(exclude_none=False)
