from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import os
import time
import numpy as np
import cv2

class DinoGame(object):

    chrome_driver_path = 'D:\workspace\chromedriver\chromedriver'
    game_url = 'chrome://dino'
    window_width = 500
    window_hight = 400
        
    def is_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def is_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def open_game(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        # chrome_options.add_argument("--start-maximized")
        self._driver = webdriver.Chrome(executable_path = self.chrome_driver_path, chrome_options=chrome_options)
        self._driver.set_window_position(x=0,y=0)
        self._driver.set_window_size(self.window_width, self.window_hight)
        self._driver.get(self.game_url)

    def start(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
        time.sleep(0.5)
        ##print("----------  Start game ---------- ")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
        time.sleep(0.5)
        ##print("----------  Restart game ---------- ")

    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
        ##print("----------  Press up ---------- ")

    def press_down(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
        ##print("----------  Press down ---------- ")

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array) # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)

    def get_screen_location(self):
        game_container_class_name = "runner-container"
        game_container_elements = self._driver.find_element_by_class_name(game_container_class_name)
        browser_navigation_panel_height = self._driver.execute_script('return window.outerHeight - window.innerHeight;')
        location = game_container_elements.location
        location['y'] = location['y'] + browser_navigation_panel_height
        size = game_container_elements.size
        return location, size

    def get_screenshort(self):
        png = self._driver.get_screenshot_as_png()
        nparr = np.frombuffer(png, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    def pause(self):
        ##print("----------  Pause game ---------- ")
        return self._driver.execute_script("return Runner.instance_.stop()")
    
    def resume(self):
        ##print("----------  Resume game ---------- ")
        return self._driver.execute_script("return Runner.instance_.play()")
    
    def end(self):
        ##print("----------  Close game ---------- ")
        self._driver.close()