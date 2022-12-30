import warnings
warnings.filterwarnings(action='ignore')

from argparse import ArgumentParser
from distutils.util import strtobool

class ExperimentArgParser(ArgumentParser):
    '''Sets up an argument parser for the command line arguments that all
    of the reinforcement learning algorithms have in common.'''
    def __init__(self, caller):
        '''Constructor'''
        super().__init__()
        self.add_argument("--exp-name", type=str, default=caller,
            help="the name of this experiment")
        self.add_argument("--seed", type=int, default=1,
            help="seed of the experiment")
        self.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="if toggled, `torch.backends.cudnn.deterministic=False`")
        self.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="if toggled, cuda will be enabled by default")
        self.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
            help="if toggled, this experiment will be tracked with Weights and Biases")
        self.add_argument("--wandb-project-name", type=str, default="cleanRL",
            help="the wandb's project name")
        self.add_argument("--wandb-entity", type=str, default=None,
            help="the entity (team) of wandb's project")
        self.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
            help="whether to capture videos of the agent performances (check out `videos` folder)")
        self.add_argument("--ignore-action-rate", type=float, default=0,
            help="a value between 0 and 1 used to determine if a chosen action should be ignored and replaced with a random action. 0.1 means random 10 percent of the time.")
        self.add_argument("--checkpoint-frequency", type=float, default=0,
            help="the number of steps to take between checkpointing the model with 0 meaning checkpointing is disabled.")
        self.add_argument("--resume", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
            help="if toggled, load the agent state from a prior run with the same experiment name from Weights and Biases")