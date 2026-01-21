from app.backtesting.training_dataset import build_training_row


def build_training_dataset(
    replay_rows: list[dict],
    records: list[dict],
    audit_result: dict,
) -> list[dict]:

    overconfidence_cases = audit_result.get("overconfidence_cases", [])

    rows = []
    for replay_row, record in zip(replay_rows, records):
        rows.append(
            build_training_row(
                replay_row=replay_row,
                record=record,
                overconfidence_cases=overconfidence_cases,
            )
        )

    return rows
