from Tkinter import *
import ImageTk
from PIL import Image
from atari_env import Environment
import numpy as np
import time



obs = np.array([])
env = Environment(rom="/home/john/atari_roms/Atari2600_A-E/Breakout.bin")
obs = [env.reset()]
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)
o, r, d = env.step(1)
obs = np.append(obs, [o], axis=0)


def callback():
    #raw_input("Press enter to continue")
    global i
    i += 1
    print(i)
    tkim = ImageTk.PhotoImage(Image.fromarray(obs[i]))
    window.configure(image=tkim)
    window.image = tkim


i=0
im1 = Image.fromarray(obs[i])
tkim = ImageTk.PhotoImage(im1)
window = Label(image=tkim)
window.image = tkim
window.pack()
# for j in range(len(obs)-1):
#     window.after(j*1000, callback)
#
# window.mainloop()

for j in range(len(obs)-1):
    raw_input("Press enter to continue")
    tkim = ImageTk.PhotoImage(Image.fromarray(obs[j]))
    window.configure(image=tkim)
    window.image = tkim
    window.update_idletasks()
    window.update()







# for i in range(3):
#     im = Image.fromarray(obs[i])
#     tkim = ImageTk.PhotoImage(im)
#     window.configure(image=tkim)
#     window.image = tkim
#     print("here")
#     window.after(1, next_frame())
