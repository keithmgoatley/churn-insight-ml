"""Generate a realistic synthetic customer dataset for churn modelling."""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 5000


def generate(path: str = "data/customers.csv") -> pd.DataFrame:
    tenure = RNG.gamma(shape=2.0, scale=12.0, size=N).clip(1, 72).round()
    monthly_spend = RNG.normal(45, 18, N).clip(5, 150).round(2)
    support_tickets = RNG.poisson(1.2, N)
    logins_per_month = RNG.gamma(2.5, 4.0, N).clip(0, 60).round()
    discount_user = RNG.binomial(1, 0.3, N)
    contract_monthly = RNG.binomial(1, 0.55, N)

    logit = (
        -1.2
        - 0.045 * tenure
        + 0.35 * support_tickets
        - 0.06 * logins_per_month
        + 0.9 * contract_monthly
        - 0.4 * discount_user
        + 0.004 * monthly_spend
    )
    p_churn = 1 / (1 + np.exp(-logit))
    churned = RNG.binomial(1, p_churn)

    df = pd.DataFrame({
        "tenure_months": tenure,
        "monthly_spend": monthly_spend,
        "support_tickets_90d": support_tickets,
        "logins_per_month": logins_per_month,
        "on_discount": discount_user,
        "monthly_contract": contract_monthly,
        "churned": churned,
    })
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = generate()
    print(f"Generated {len(df)} customers | churn rate: {df.churned.mean():.1%}")
