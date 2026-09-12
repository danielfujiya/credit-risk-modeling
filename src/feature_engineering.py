from pyspark.sql import DataFrame, functions as F


VALID_RANGES = {
    "Num_Bank_Accounts": (0, 50),
    "Num_Credit_Card": (0, 50),
    "Interest_Rate": (0, 100),
    "Num_of_Loan": (0, 20),
}

LOAN_TYPE_FEATURES = {
    "Auto Loan": "has_auto_loan",
    "Credit-Builder Loan": "has_credit_builder_loan",
    "Debt Consolidation Loan": "has_debt_consolidation_loan",
    "Home Equity Loan": "has_home_equity_loan",
    "Mortgage Loan": "has_mortgage_loan",
    "Not Specified": "has_not_specified",
    "Payday Loan": "has_payday_loan",
    "Personal Loan": "has_personal_loan",
    "Student Loan": "has_student_loan",
}

LEAKAGE_RISK_COLUMNS = [
    "Delay_from_due_date",
    "Payment_Behaviour",
    "Payment_of_Min_Amount",
]


def prepare_model_input(source_df: DataFrame) -> DataFrame:
    """Aplica transformações determinísticas usadas pelo modelo de PD."""

    result_df = source_df

    for column, (lower_bound, upper_bound) in VALID_RANGES.items():
        result_df = result_df.withColumn(
            column,
            F.when(
                F.col(column).between(lower_bound, upper_bound),
                F.col(column),
            ).otherwise(F.lit(None)),
        )

    result_df = result_df.withColumn(
        "Credit_Mix",
        F.when(
            F.trim(F.col("Credit_Mix")).isin("-", "_"),
            F.lit(None),
        ).otherwise(F.trim(F.col("Credit_Mix"))),
    )

    result_df = result_df.withColumn(
        "Occupation",
        F.trim(F.col("Occupation")),
    )

    loan_types_array = F.split(
        F.regexp_replace(
            F.coalesce(F.col("Type_of_Loan"), F.lit("")),
            r",\s*and\s+",
            ",",
        ),
        r"\s*,\s*",
    )

    for loan_type, feature_name in LOAN_TYPE_FEATURES.items():
        result_df = result_df.withColumn(
            feature_name,
            F.array_contains(loan_types_array, loan_type).cast("int"),
        )

    return result_df.drop(
        "Type_of_Loan",
        *LEAKAGE_RISK_COLUMNS,
    )