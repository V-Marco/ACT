from datetime import datetime


class ACTDummyLogger:
    def __init__(self):
        pass

    def info(self, msg: str):
        pass

    def epoch(self, epoch_num: int, loss_name: str, loss_value: float):
        pass


class ACTLogger:
    def __init__(self):
        print(f"\n({datetime.now()})-[START]")

    def info(self, msg: str) -> None:
        print(f"({datetime.now()})-[INFO]: {msg}")

    def epoch(self, epoch_num: int, loss_name: str, loss_value: float) -> None:
        print(
            f"({datetime.now()})-[EPOCH {epoch_num}]: {loss_name} = {round(loss_value, 5)}"
        )
