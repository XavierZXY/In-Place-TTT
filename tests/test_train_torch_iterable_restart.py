import pytest

from tasks.train_torch import _next_train_micro_batches


class RestartableDataloader:
    def __init__(self):
        self.epoch = 0
        self.set_epochs = []
        self.batches_by_epoch = {
            0: [],
            1: [["batch-from-next-epoch"]],
        }

    def __iter__(self):
        return iter(self.batches_by_epoch.get(self.epoch, []))

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.set_epochs.append(epoch)


def test_next_train_micro_batches_restarts_iterable_dataloader_after_exhaustion():
    dataloader = RestartableDataloader()
    data_iterator = iter([])

    micro_batches, data_iterator, stream_epoch, restarted = _next_train_micro_batches(
        dataloader,
        data_iterator,
        restart_on_stop_iteration=True,
        stream_epoch=0,
    )

    assert micro_batches == ["batch-from-next-epoch"]
    assert list(data_iterator) == []
    assert stream_epoch == 1
    assert restarted is True
    assert dataloader.set_epochs == [1]


def test_next_train_micro_batches_keeps_finite_dataloader_exhaustion_behavior():
    dataloader = RestartableDataloader()
    data_iterator = iter([])

    with pytest.raises(StopIteration):
        _next_train_micro_batches(
            dataloader,
            data_iterator,
            restart_on_stop_iteration=False,
            stream_epoch=0,
        )
