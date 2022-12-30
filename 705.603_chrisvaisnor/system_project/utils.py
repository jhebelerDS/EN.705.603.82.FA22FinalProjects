'''Holds the score function and the get_device function'''
import torch

def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return score: the score of the prediction
    """
    return int(true_expansion == pred_expansion)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
