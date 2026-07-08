import operator
import pickle
from NeuralNet import FeedForward, tanh, sigmoid, softmax
import random
import numpy
import time

# TODO:apply softmax to for weightage
#TODO: hashing individuals with inheriting numbers?
#TODO:Saving consistent bests

init_size = 50000
pop_siz = 500
crossover_par = 25
mutant_par = 10

elitism = 50
mutants = 430
crossover = 10
new_ind = 10

mut_factor = 0.1
runs_per_ind = 1

num_gen = 10000
save_every = 5

inputs = 8
hl_1 = 6
hl_2 = 6
outputs = 2


class Individual:
    def __init__(self):
        self.brain = FeedForward(inputs)
        self.brain.add_layer(hl_1, tanh)
        self.brain.add_layer(hl_2, tanh)
        self.brain.add_layer(outputs, sigmoid)

        self.fitness = 0
        self.tag = 'N'  #New Individual

    def set_wts(self, wts):
        for z in range(len(wts)):
            self.brain.layers[z].weights = wts[z].copy()

    def get_wts(self):
        wt1 = []
        for l in self.brain.layers:
            x = numpy.copy(l.weights)
            wt1.append(x)

        return wt1


'''The function to be given to the genetic algorithm class should take an Individual class object as a parameter which it
should then test and assign a fitness to the individual object
The fitness should be always non zero and postive'''


class GeneticAlg:
    def __init__(self, func, env=None):
        self.winner = None
        self.env = env
        self.members = []
        for x in range(init_size):
            self.members.append(Individual())
        self.func = func
        self.num_gen = num_gen
        self.gen_n = 1
        self.generations = []
        self.best_fitnesses = []
        self.top_10_fitness = []
        self.avg_fitnesses = []
        self.new_m = []
        self.eli_m = []
        self.mut_m = []
        self.cro_m = []
        self.new_a = []
        self.eli_a = []
        self.mut_a = []
        self.cro_a = []

    def create_new_individual_crossover(self, p1: Individual, p2: Individual):
        wt1 = p1.get_wts()
        wt2 = p2.get_wts()

        new_wt = []

        for z in range(len(wt1)):
            x1 = wt1[z]
            p = []
            for i in range(len(x1)):
                q = []
                for j in range(len(x1[0])):
                    q.append(random.choice([wt1[z][i][j], wt2[z][i][j]]))
                p.append(q)

            p = numpy.array(p, dtype='float64')
            new_wt.append(p)

        new_ind = Individual()
        new_ind.set_wts(new_wt)
        return new_ind

    def mutate_wts(self, wt):
        wts = wt.copy()
        w = []
        for z in range(len(wts)):
            shape = wts[z].shape
            x = numpy.random.randn(shape[0], shape[1]) * mut_factor
            a = numpy.copy(wts[z]) + x
            w.append(a)
        return w

    def arrange_based_on_fitness(self):
        self.members.sort(key=operator.attrgetter('fitness'))

    def get_fittest(self, n, new=True):
        self.arrange_based_on_fitness()
        self.members.reverse()
        el = []
        for x in range(n):
            ind = self.members[x]
            wts = ind.get_wts()
            if new:
                new_i = Individual()
                new_i.set_wts(wts)
                el.append(new_i)
            else:
                el.append(ind)

        return el

    def create_mut_ind(self, parent: Individual):
        wts = parent.get_wts()
        new_wts = self.mutate_wts(wts)
        ind = Individual()
        ind.set_wts(new_wts)
        return ind

    def choose_parents(self):
        self.arrange_based_on_fitness()
        par = []
        weights = []
        for x in range(crossover_par):
            par.append(self.members[x])
            weights.append(self.members[x].fitness)
        weights = softmax(weights)
        x = random.choices(par, weights, k=2)
        return x

    def create_next_gen(self):
        new_pop = []
        #New Random Members
        for x in range(new_ind):
            new_pop.append(Individual())
        #ELitist Members
        elitists = self.get_fittest(elitism)
        for ind in elitists:
            ind.tag = 'E'
            new_pop.append(ind)
        #Mutant Members
        mutant_parents = self.get_fittest(mutant_par)
        for x in range(mutants):
            ind = self.create_mut_ind(random.choice(mutant_parents))
            ind.tag = 'M'
            new_pop.append(ind)
        #Crossover_members
        for x in range(crossover):
            par = self.choose_parents()
            ind = self.create_new_individual_crossover(par[0], par[1])
            ind.tag = 'C'
            new_pop.append(ind)

        return new_pop

    def run_pop(self):
        for ind in self.members:
            for i in range(runs_per_ind):
                self.func(ind)
            ind.fitness /= runs_per_ind

    def print_data(self):
        avg_fitness = 0
        for member in self.members:
            avg_fitness += member.fitness
        avg_fitness /= pop_siz

        best_10 = self.get_fittest(10, new=False)
        top_10_avg = 0
        for member in best_10:
            top_10_avg += member.fitness
        top_10_avg /= 10
        best_fitness = self.get_fittest(1, new=False)[0].fitness
        best_fit_tag = self.get_fittest(1, new=False)[0].tag
        print(f'######  Generation Number {self.gen_n}  ######')
        print(f'Best Fitness = {best_fitness}')
        print(f'Top 10 Average = {top_10_avg}')
        print(f'Average Fitness = {avg_fitness}')
        print(f'''This generation's fittest is {best_fit_tag}''')
        self.generations.append(self.gen_n)
        self.best_fitnesses.append(best_fitness)
        self.top_10_fitness.append(top_10_avg)
        self.avg_fitnesses.append(avg_fitness)
        self.avg_fitnesses[0] = 0
        self.individual_data()
        graphing_stuff = [self.generations, self.best_fitnesses, self.top_10_fitness, self.avg_fitnesses,
                          self.mut_m, self.cro_m, self.eli_m, self.new_m,
                          self.mut_a, self.cro_a, self.eli_a, self.new_a]
        with open('GraphingStuff.pickle', 'wb') as f:
            pickle.dump(graphing_stuff, f)
            f.close()

    def run_alg(self):
        print('Starting Training...')
        print('Assessing Initial Population...')
        for x in range(self.num_gen):
            t1 = time.time()
            self.run_pop()
            self.winner = self.get_fittest(1)[0]
            self.print_data()
            new_gen = self.create_next_gen()
            if self.gen_n % save_every == 0 and self.gen_n != 0:
                self.save_winner('SavedWinner.pickle')
                self.save_gen('SavedGeneration.pickle')
                self.save_top_10('SavedTop10.pickle')
                print('Checkpoint Saved')

            self.members = new_gen

            self.save_winner('Fittest.pickle')
            t2 = time.time()
            print(f'Runtime: {t2 - t1}sec')
            if self.env:
                self.env.next_gen()
            self.gen_n += 1
        self.save_winner('SavedWinner.pickle')
        self.save_gen('SavedGeneration.pickle')
        self.save_top_10('SavedTop10.pickle')
        print('Algorithm Done')

    def return_winner(self):
        return self.winner

    def save_gen(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self.members, file)

    def load_gen(self, filepath):
        with open(filepath, 'rb') as file:
            self.members = pickle.load(file)

    def save_winner(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self.return_winner(), file)

    def save_top_10(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self.get_fittest(10), file)

    def individual_data(self):
        eli = []
        mut = []
        new = []
        cro = []
        for ind in self.members:
            if ind.tag == 'N':
                new.append(ind.fitness)
            elif ind.tag == 'E':
                eli.append(ind.fitness)
            elif ind.tag == 'M':
                mut.append(ind.fitness)
            elif ind.tag == 'C':
                cro.append(ind.fitness)
        if len(eli) != 0:
            e_a = 0
            e_m = 0
            for x in eli:
                e_a += x
                if x > e_m:
                    e_m = x
            e_a /= len(eli)
            self.eli_m.append(e_m)
            self.eli_a.append(e_a)
        else:
            self.eli_m.append(0)
            self.eli_a.append(0)

        if len(new) != 0:
            n_a = 0
            n_m = 0
            for x in new:
                n_a += x
                if x > n_m:
                    n_m = x
            n_a /= len(new)
            self.new_m.append(n_m)
            self.new_a.append(n_a)
        else:
            self.new_m.append(0)
            self.new_a.append(0)
        if len(mut) != 0:
            m_a = 0
            m_m = 0
            for x in mut:
                m_a += x
                if x > m_m:
                    m_m = x
            m_a /= len(mut)
            self.mut_m.append(m_m)
            self.mut_a.append(m_a)
        else:
            self.mut_m.append(0)
            self.mut_a.append(0)
        if len(cro) != 0:
            c_a = 0
            c_m = 0
            for x in cro:
                c_a += x
                if x > c_m:
                    c_m = x
            c_a /= len(cro)
            self.cro_m.append(c_m)
            self.cro_a.append(c_a)
        else:
            self.cro_m.append(0)
            self.cro_a.append(0)

    def create_gen_from_ind(self, filepath):
        with open(filepath, 'rb') as file:
            ind = pickle.load(file)
        new_pop = [ind]
        for x in range(pop_siz - 1):
            indn = self.create_mut_ind(ind)
            new_pop.append(indn)

        return new_pop

    def create_gen_from_group(self, filepath):
        with open(filepath, 'rb') as file:
            inds = pickle.load(file)
        new_pop = []
        for x in range(pop_siz - len(inds)):
            indn = self.create_mut_ind(random.choice(inds))
            new_pop.append(indn)
        for i in inds:
            new_pop.append(i)

        return new_pop
