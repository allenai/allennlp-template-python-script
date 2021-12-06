# This file contains a bunch of build_* methods that configure objects however you want, and a
# run_training_loop method that calls these methods and runs the trainer.

from itertools import chain
from typing import Iterable, Tuple

import allennlp
import torch
from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.trainer import Trainer
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer

from my_project.dataset_reader import ClassificationTsvReader
from my_project.model import SimpleClassifier


def build_dataset_reader() -> DatasetReader:
    return ClassificationTsvReader()


def build_vocab(train_loader, dev_loader) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(
        chain(train_loader.iter_instances(), dev_loader.iter_instances())
    )


def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)}
    )
    encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
    return SimpleClassifier(vocab, embedder, encoder)


def build_data_loaders(
    reader,
    train_data_path: str,
    validation_data_path: str,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = MultiProcessDataLoader(
        reader, train_data_path, batch_size=8, shuffle=True
    )
    dev_loader = MultiProcessDataLoader(
        reader, validation_data_path, batch_size=8, shuffle=False
    )
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)  # type: ignore
    # There are a *lot* of other things you could configure with the trainer.  See
    # http://docs.allennlp.org/master/api/training/trainer/#gradientdescenttrainer-objects for more
    # information.

    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
        validation_metric="+accuracy",
    )
    return trainer


def run_training_loop(serialization_dir: str):
    reader = build_dataset_reader()

    train_loader, dev_loader = build_data_loaders(
        reader, "/path/to/your/training/data", "/path/to/your/validation/data"
    )

    vocab = build_vocab(train_loader, dev_loader)
    model = build_model(vocab)

    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)

    # NOTE: Training using multiple GPUs is hard in this setting.  If you want multi-GPU training,
    # we recommend using our config file template instead, which handles this case better, as well
    # as saving the model in a way that it can be easily loaded later.  If you really want to use
    # your own python script with distributed training, have a look at the code for the allennlp
    # train command (https://github.com/allenai/allennlp/blob/master/allennlp/commands/train.py),
    # which is where we handle distributed training.  Also, let us know on github that you want
    # this; we could refactor things to make this usage much easier, if there's enough interest.

    print("Starting training")
    trainer.train()
    print("Finished training")
