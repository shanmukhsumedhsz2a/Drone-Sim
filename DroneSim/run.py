from main import DroneSim
from GeneticAlg import GeneticAlg

sim = DroneSim(seed=2)
trainer = GeneticAlg(sim.evaluate_ind, env=sim)
trainer.load_gen('SavedGeneration.pickle')
for ind in trainer.members:
    ind.tag = 'E'
trainer.run_alg()
win = trainer.return_winner()

sim2 = DroneSim()
for i in range(100):
    sim2.run_ind(win)