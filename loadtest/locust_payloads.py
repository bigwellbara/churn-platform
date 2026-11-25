"""Random but always-valid customer payloads, shared by the load test and the profiler."""

import random

PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def random_customer() -> dict:
    internet = random.choices(["DSL", "Fiber optic", "No"], weights=[34, 44, 22])[0]
    phone = random.choices(["Yes", "No"], weights=[90, 10])[0]
    tenure = random.randint(0, 72)
    monthly = round(random.uniform(18.25, 118.75), 2)

    def addon() -> str:
        return "No internet service" if internet == "No" else random.choice(["Yes", "No"])

    return {
        "gender": random.choice(["Male", "Female"]),
        "SeniorCitizen": random.choices([0, 1], weights=[84, 16])[0],
        "Partner": random.choice(["Yes", "No"]),
        "Dependents": random.choices(["Yes", "No"], weights=[30, 70])[0],
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": "No phone service" if phone == "No" else random.choice(["Yes", "No"]),
        "InternetService": internet,
        "OnlineSecurity": addon(),
        "OnlineBackup": addon(),
        "DeviceProtection": addon(),
        "TechSupport": addon(),
        "StreamingTV": addon(),
        "StreamingMovies": addon(),
        "Contract": random.choices(["Month-to-month", "One year", "Two year"], [55, 21, 24])[0],
        "PaperlessBilling": random.choice(["Yes", "No"]),
        "PaymentMethod": random.choice(PAYMENT_METHODS),
        "MonthlyCharges": monthly,
        "TotalCharges": round(monthly * tenure, 2),
    }
