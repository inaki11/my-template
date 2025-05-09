class BaseCallback:
    def on_train_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_validation_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_batch_end(self, trainer):
        pass
