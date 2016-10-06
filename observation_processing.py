from PIL import Image
import numpy as np


def preprocess(obs):
    img = Image.fromarray(obs)
    img = img.resize((84,84))
    new_obs = np.array(img)/255.
    return new_obs


def blacken_score(obs, row, col):
    for i in range(0, int(.1*row)):
        for j in range(col):
            obs[i][j] = 0.0

    return obs
