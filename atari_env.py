from ale_python_interface import ALEInterface
import numpy as np
from Tkinter import *
import ImageTk
from PIL import Image
import time
import random



class Environment:

    def __init__(self, rom):
        self.ale = ALEInterface()
        self.ale.loadROM(rom_file=rom)
        self.action_space = self.ale.getMinimalActionSet()
        self.obs = self.reset()
        self.im = Image.fromarray(self.obs)
        self.root = Tk()
        self.tkim = ImageTk.PhotoImage(self.im)
        self.window = Label(image=self.tkim)
        self.window.image = self.tkim
        self.window.pack()


    def step(self, action):
        reward = 0

        for i in range(4):
            reward += self.ale.act(self.action_space[action])
            if i == 2:
                frame1 = self.ale.getScreenGrayscale()
            if i == 3:
                frame2 = self.ale.getScreenGrayscale()

        self.obs = np.squeeze(np.maximum(frame1, frame2))
        done = self.ale.game_over()
        return self.obs, reward, done

    def reset(self):
        self.ale.reset_game()
        self.obs = np.squeeze(self.ale.getScreenGrayscale())
        return self.obs

    def render(self, rate=0.1):
        self.im = Image.fromarray(self.obs)
        self.tkim = ImageTk.PhotoImage(self.im)
        self.window.configure(image=self.tkim)
        self.window.image = self.tkim
        self.window.update_idletasks()
        self.window.update()
        time.sleep(rate)

    def sample_action(self):
        action = random.choice([0, 1, 2, 3])
        return action

