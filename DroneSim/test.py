from main import DroneSim
import pickle

win = None
with open('Trained.pickle', 'rb') as file:
    top = [pickle.load(file)]

sim = DroneSim(display=True, interactive=True)
sim.frames_per_target = 50
sim.target_rad = 15

for x in range(100):
    sim.run_ind(top[0])
    sim.next_gen()
