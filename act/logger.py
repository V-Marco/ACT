from datetime import datetime


class ACTDummyLogger:
    @staticmethod
    def info(msg: str): pass
    @staticmethod
    def log(msg: str): pass
    @staticmethod
    def epoch(epoch_num: int, loss_name: str, loss_value: float): pass


class ACTLogger:

    @staticmethod
    def info(msg: str) -> None:
        print(f"({datetime.now()})-[INFO]: {msg}")
    
    @staticmethod
    def log(msg: str) -> None:
        ACTLogger.info(msg)

    @staticmethod
    def epoch(epoch_num: int, loss_name: str, loss_value: float) -> None:
        print(
            f"({datetime.now()})-[EPOCH {epoch_num}]: {loss_name} = {round(loss_value, 5)}"
        )
