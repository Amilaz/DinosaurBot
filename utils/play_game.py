from utils.dino_game import DinoGame
from utils.game_state import GameSate
from utils import model as model_utils
from utils import const
from utils import training

def play_game(observe=False):
    game = DinoGame()
    game_state = GameSate(game)
    model = model_utils.build_model(const.ACTIONS, const.INPUT_SIZE)
    training.train_model(model, game_state)