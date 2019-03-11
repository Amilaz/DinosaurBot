from utils import image_utils
import time

class GameSate:
    def __init__(self, game):
        self._game = game
        self._game.open_game()
        
    def initial_image_state(self):
        self._game.start()
        # image_list = []
        # time.sleep(0.1)
        # for _ in range(3):
        #     time.sleep(0.15)
        #     # Screen capture
        #     location, size = self._game.get_screen_location()
        #     driver_image = self._game.get_screenshort()
        #     image = image_utils.grab_screen(driver_image, location['x'], location['y'], size['height'], self._game.window_width)
        #     image_list.append(image)
        # return image_list

    def pause_game(self):
        self._game.pause()

    def resume_game(self):
        self._game.resume()

    def get_state(self, actions, image_sample=False):
        score = self._game.get_score() 
        is_over = False
        # Press up event
        # reward = score / 10
        if actions[1] == 1:
            self._game.press_up()
            reward = score / 10
        else:
            reward = score / 7
        # Screen capture
        location, size = self._game.get_screen_location()
        driver_image = self._game.get_screenshort()
        image = image_utils.grab_screen(driver_image, location['x'], location['y'], size['height'], self._game.window_width, image_sample=image_sample)
        # Game crashed event
        if self._game.is_crashed():
            self._game.restart()
            if actions[1] == 1:
                reward = 1 / score
            else:
                reward = 10 / score
            is_over = True
        return image, reward, score, is_over