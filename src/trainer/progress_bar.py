from rich.progress import Progress
from typing import Optional, Union
from src.trainer.callback import Callback


class ProgressBar(Callback):
    """
    Class stub for type checking on Trainin initialization.
    """
    pass


class RichProgressBar(ProgressBar):
    def __init__(self, trainer: "Trainer", refresh_rate: int = 1) -> None:  # type: ignore
        super().__init__()
        self.refresh_rate = refresh_rate
        self.train_task_id = None
        self.validation_task_id = None
        self.test_task_bar_id = None
        self.progress = Progress(auto_refresh=False)
        self.is_enabled = True
        self._owner = trainer

    def train_init(self, *_):
        self.train_task_id = self.progress.add_task(
            "Training loop progress bar",
            start=True,
            total=self._owner.eval_freq,
        )

    def train_iter_end(self, i, *_):
        if self.train_task_id is not None:
            self._update(self.train_task_id, i)

    def train_end(self, total_steps, *_):
        if self.train_task_id is not None:
            self._update(self.train_task_id, total_steps)
            self.train_task_id = None
            self.progress.remove_task(self.train_task_id)  # type: ignore
        self.progress.refresh()

    def validation_start(self, *_):
        self.validation_task_id = self.progress.add_task(
            "Validation",
            start=True,
            total=self._owner.eval_steps,
        )

    def validation_iter_end(self, i, *_):
        if self.train_task_id is not None:
            self._update(self.validation_task_id, i)

    # def _add_task(self,

    # def _get_train_description(self, current_epoch: int) -> str:
    #         train_description = f"Epoch {current_epoch}"
    #         if self._owner.steps is not None:
    #             train_description += f"/{self.trainer.max_epochs - 1}"
    #         if len(self.validation_description) > len(train_description):
    #             # Padding is required to avoid flickering due of uneven lengths of "Epoch X"
    #             # and "Validation" Bar description
    #             train_description = f"{train_description:{len(self.validation_description)}}"
    #         return train_description

    def _update(
        self,
        progress_bar_id: Optional["TaskID"],  # type: ignore
        current: int,
        visible: bool = True
    ) -> None:
        if self.progress is not None and self.is_enabled:
            assert progress_bar_id is not None
            total = self.progress.tasks[progress_bar_id].total
            assert total is not None
            if not self._should_update(current, total):
                return

            leftover = current % self.refresh_rate
            advance = leftover if (current == total and leftover != 0) else self.refresh_rate
            self.progress.update(progress_bar_id, advance=advance, visible=visible)
            self.progress.refresh()

    def _should_update(self, current: int, total: Union[int, float]) -> bool:
        return current % self.refresh_rate == 0 or current == total


if __name__ == "__main__":
    class Trainer:
        def __init__(self) -> None:
           self.steps = 10
           self.eval_steps = 10
           self.eval_freq = 2

    # progress_bar = RichProgressBark(Trainer() ej
