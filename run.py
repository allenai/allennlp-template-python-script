# There are lots of way to set up this training script.  We're putting the bulk of the code inside
# the my_project module, with a simple run script in the base directory.  If you prefer, you could
# just take train.py and move it to the top-level directory and use that as your run.py.  Do
# whatever you're most comfortable with.

from my_project.train import run_training_loop

run_training_loop(serialization_dir="results/")
