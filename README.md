# AllenNLP Template Project using your own python script

A template for starting a new allennlp project using your own python script, instead of config files
and `allennlp train`.  For simple projects, all you need to do is get your model code into the class
in `my_project/model.py`, your data loading code into the `DatasetReader` code in
`my_project/dataset_reader.py`, and configuration code in `my_project/train.py` (the `build_*`
methods), and that's it, you can train your model with `python run.py`.  We recommend also making
appropriate changes to the test code, and using that for development, but that's optional.

See the [AllenNLP Guide](https://guide.allennlp.org/your-first-model) for a quick start on how to
use what's in this example project.  We're grabbing the model and dataset reader classes from that
guide.  You can replace those classes with a model and dataset reader for whatever you want
(including copying code from our [model library](https://github.com/allenai/allennlp-models) as a
starting point). The very brief version of what's in here:

* A `Model` class in `my_project/model.py`.
* A `DatasetReader` class in `my_project/dataset_reader.py`.
* Tests for both of these classes in `tests`, including a small toy dataset that can be read.  We
  strongly recommend that you use a toy dataset with tests like this during model development, for
  quick debug cycles. To run the tests, just run `pytest` from the base directory of this project.
* A script to configure the model, dataset reader, and other training loop objects, in
  `my_project/train.py`.  The `build_*` methods are meant to be changed according to however you
  want to setup your training run.  You probably don't need to change the `run_training_loop`
  function.  To train the model just run `python run.py` after doing `pip install allennlp`.
