import wandb
import time
import torch
from torch.utils.tensorboard import SummaryWriter

def construct_tracking_name_for_run(params):
    '''Creates and returns the name to use for tracking the run within WandB and Tensorboard.'''
    return f"{params.env_id}__{params.exp_name}__{params.seed}__{int(time.time())}"
    # return f"{params.env_id}__{params.exp_name}__{params.seed}"

def intialize_wandb(run_name, params):
    '''Sets up tracking with weights and biases. Arguments include the wandb_project_name,
    wandb_entity, and run_name.'''
    run = wandb.init(
        project=params.wandb_project_name,
        entity=params.wandb_entity,
        sync_tensorboard=True,
        config=vars(params),
        name=run_name,
        monitor_gym=True,
        save_code=True,
        resume=params.resume
    )
    return run

def construct_tensorboard_writer(run_name, params):
    '''Constructs and returns SummaryWriter for tracking metrics using Tensorboard.'''
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(params).items()])),
    )
    return writer

def checkpoint_model(agent):
    '''Saves the model to a file and optionally saves it as an artifact in WandB.'''
    print('Saving model checkpoint')
    torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
    wandb.save(f"{wandb.run.dir}/agent.pt", policy="now")

def resume_run_if_requested(run, agent, device):
    '''Resumes training on a saved model if requested via WandB environment variables.'''
    if (wandb.run.resumed):
        print('Resuming prior run.')
        filename = wandb.restore("agent.pt").name
        print(f'Attempting to load file {filename}')
        agent.load_state_dict(torch.load(filename, map_location=device))
        agent.eval()
        global_step = run.summary.get("global_step") + 1
        return agent, global_step