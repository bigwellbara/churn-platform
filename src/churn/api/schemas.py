"""Request and response models. FastAPI validates every request against these before
the model sees it, and generates the OpenAPI docs (/docs) from them."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]

INTERNET_ADDONS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

EXAMPLE_CUSTOMER = {
    "customer_id": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


class CustomerFeatures(BaseModel):
    """One customer. Field names match the training data columns exactly."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    customer_id: str | None = Field(
        default=None, max_length=64, description="Your identifier; echoed back, not a feature"
    )
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=1_000)
    TotalCharges: float = Field(ge=0, le=100_000)

    @model_validator(mode="after")
    def check_consistency(self) -> CustomerFeatures:
        """Reject combinations that cannot happen in the real data. These indicate a bug
        in the calling system, and the model would return a confident but meaningless score."""
        if (self.PhoneService == "No") != (self.MultipleLines == "No phone service"):
            raise ValueError(
                "MultipleLines must be 'No phone service' exactly when PhoneService is 'No'"
            )
        no_internet = self.InternetService == "No"
        for field in INTERNET_ADDONS:
            if (getattr(self, field) == "No internet service") != no_internet:
                raise ValueError(
                    f"{field} must be 'No internet service' exactly when InternetService is 'No'"
                )
        return self

    def features(self) -> dict:
        """Model inputs only (drops customer_id)."""
        return self.model_dump(exclude={"customer_id"})


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[CustomerFeatures] = Field(min_length=1)


class Prediction(BaseModel):
    customer_id: str | None
    churn_probability: float = Field(description="Model score in [0, 1]; use for ranking")
    churn_prediction: bool = Field(description="score >= decision threshold")
    risk_level: Literal["low", "medium", "high"]


class PredictionResponse(Prediction):
    model_config = ConfigDict(protected_namespaces=())
    model_version: str
    threshold: float


class BatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_version: str
    threshold: float
    count: int
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: Literal["ok"]
    model_version: str
    database: Literal["connected", "disabled", "unavailable"]


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    model_version: str
    model_type: str
    decision_threshold: float
    exported_at: str | None = None
    features: list[str]
    test_metrics: dict[str, float]


class ErrorResponse(BaseModel):
    error: str
    detail: list[dict] | str
    request_id: str | None = None
