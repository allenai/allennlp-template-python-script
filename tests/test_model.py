import tempfile

from allennlp.common.testing import ModelTestCase

from my_project.train import (
    build_dataset_reader,
    build_vocab,
    build_model,
    build_data_loaders,
    build_trainer,
)


class TestSimpleClassifier(ModelTestCase):
    def test_model_can_train(self):
        with tempfile.TemporaryDirectory() as serialization_dir:
            reader = build_dataset_reader()
            train_loader, dev_loader = build_data_loaders(
                reader, "tests/fixtures/toy_data.tsv", "tests/fixtures/toy_data.tsv"
            )
            vocab = build_vocab(train_loader, dev_loader)
            train_loader.index_with(vocab)
            dev_loader.index_with(vocab)
            # Ideally you'd want to build a tiny toy model here, instead of calling the full
            # build_model function, like we do with the data above.
            model = build_model(vocab)
            trainer = build_trainer(
                model, serialization_dir, train_loader, train_loader
            )
            # This built-in test makes sure that your data can load, that it gets passed to the
            # model correctly, that your model computes a loss in a way that we can get gradients
            # from it, and that all of your parameters get non-zero gradient updates.
            self.ensure_model_can_train(trainer)
