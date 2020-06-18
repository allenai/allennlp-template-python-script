# This file contains a bunch of build_* methods that configure objects however you want, and a
# run_training_loop method that calls these methods and runs the trainer.

from typing import Iterable, Tuple

import allennlp
import torch
from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer

from my_project.dataset_reader import ClassificationTsvReader
from my_project.model import SimpleClassifier


def build_dataset_reader() -> DatasetReader:
    return ClassificationTsvReader()


def read_data(
    reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    training_data = reader.read("/path/to/your/training/data")
    validation_data = reader.read("/path/to/your/validation/data")
    return training_data, validation_data


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
    return SimpleClassifier(vocab, embedder, encoder)


def build_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset,
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=8, shuffle=False)
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader
) -> Trainer:
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = AdamOptimizer(parameters)
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
    dataset_reader = build_dataset_reader()

    # These are a subclass of pytorch Datasets, with some allennlp-specific
    # functionality added.
    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)

    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    # These are again a subclass of pytorch DataLoaders, with an
    # allennlp-specific collate function, that runs our indexing and
    # batching code.
    train_loader, dev_loader = build_data_loaders(train_data, dev_data)

    trainer = build_trainer(
        model,
        serialization_dir,
        train_loader,
        dev_loader
    )

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
