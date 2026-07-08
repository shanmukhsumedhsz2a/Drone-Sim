import random
import numpy as np
import math
import time
import turtle
from GeneticAlg import Individual


def map(x, xmin, xmax, map1, map2):
    span1 = xmax - xmin
    span2 = map2 - map1

    w = (x - xmin) * span2 / span1

    out = map1 + w
    return out


class DroneSim:
    def __init__(self, manual=False,display=False,seed=0,interactive=False):
        self.manual = manual
        self.display = display
        self.interactive = interactive
        self.seed = seed

        self.wn = turtle.Screen()
        self.w = 1000
        self.h = 800
        self.wn.setup(self.w, self.h, 0)
        self.wn.title('DroneSim')
        self.wn.bgcolor('black')
        self.wn.tracer(0)
        if self.manual:
            turtle.listen()
            turtle.onkey(self.up, 'Up')
            turtle.onkey(self.down, 'Down')
            turtle.onkey(self.left, 'Left')
            turtle.onkey(self.right, 'Right')
        if self.interactive:
            self.wn.onclick(self.relocate_to_mouse)

        self.pen = turtle.Turtle()
        self.pen.penup()
        self.pen.hideturtle()

        self.pos = np.array([0, 0], dtype=np.float64)
        self.vel = np.array([0, 0], dtype=np.float64)
        self.acc = np.array([0, 0], dtype=np.float64)

        self.theta = 0
        self.ang_vel = 0
        self.ang_acc = 0
        # Parameters Of Simulation
        if self.manual:
            self.grav = 0.4
            self.air_friction = 0.965
            self.rot_friction = 0.95
            self.max_thrust = 1
            self.thrust_default = 0.2
            self.mass = 1.8
            self.length = 5
            self.rot_default = 0.6

        else:
            self.grav = 0.4
            self.air_friction = 0.97
            self.rot_friction = 0.95
            self.max_thrust = 1
            self.thrust_default = 0
            self.mass = 0.8
            self.length = 5
            self.rot_default = 0.6

        self.thrust1 = self.thrust_default
        self.thrust1_angle = 0
        self.thrust2 = self.thrust_default
        self.thrust2_angle = 0

        self.wn = turtle.Screen()
        self.w = 1000
        self.h = 800
        self.wn.setup(self.w, self.h, 0)
        self.wn.title('Drone Simulation')
        self.wn.bgcolor('black')
        self.wn.tracer(0)

        if self.interactive:
            turtle.listen()
            turtle.onkey(self.up, 'Up')
            turtle.onkey(self.down, 'Down')
            turtle.onkey(self.left, 'Left')
            turtle.onkey(self.right, 'Right')
            self.wn.onclick(self.relocate_to_mouse)

        self.pen = turtle.Turtle()
        self.pen.penup()
        self.pen.hideturtle()
        self.pen.speed(0)

        x = int(self.w / 2) - 100
        y = int(self.h / 2) - 100
        self.targets = [[220, 150], [0, 270], [-180, -175], [-40, 230], [30, -75], [330, -260], [250, 300], [-290, 300],
                        [-120, -200], [-20, 200], [260, -180], [280, 300],[0,100],[255,-255],[-400,200],[0,-200]]
        self.score = 0
        self.target_ctr = seed
        self.target = np.array([random.randint(-x, x), random.randint(-y, y)], dtype=np.float64)
        self.target = self.targets[self.target_ctr]

        self.target_rad = 10
        self.target_frames = 100
        self.target_frame_ctr = 100
        self.running = True
        self.frames_left = 750 #Divide by 50 for seconds
        self.frames_per_target = 750
        self.reward = 0
        self.gen_n = 0

    def render(self):
        self.pen.clear()

        self.pen.goto(float(self.pos[0]), float(self.pos[1]))
        self.pen.shape('circle')
        self.pen.shapesize(1.5, 1.5)
        self.pen.setheading(self.theta)
        self.pen.color('Blue')
        self.pen.stamp()

        self.pen.goto(float(self.pos[0]), float(self.pos[1]))
        self.pen.shape('square')
        self.pen.shapesize(1, 3)
        self.pen.setheading(self.theta)
        self.pen.color('Blue')
        self.pen.stamp()

        x1 = map(self.thrust1, 0, self.max_thrust + self.thrust_default, 3, 25)

        booster_pos1 = self.pos + 30 * np.array(
            [math.cos(self.theta * math.pi / 180), math.sin(self.theta * math.pi / 180)],
            dtype='float64') - x1 * np.array(
            [math.cos((self.theta + 90) * math.pi / 180), math.sin((self.theta + 90) * math.pi / 180)],
            dtype='float64')

        self.pen.goto(float(booster_pos1[0]), float(booster_pos1[1]))
        self.pen.shape('triangle')
        self.pen.shapesize(0.6, 1.5)
        self.pen.setheading(self.theta + self.thrust2_angle - 90)
        self.pen.color('orange')
        self.pen.stamp()

        thruster_pos1 = self.pos + 30 * np.array(
            [math.cos(self.theta * math.pi / 180), math.sin(self.theta * math.pi / 180)],
            dtype='float64') - 5 * np.array(
            [math.cos((self.theta + 90) * math.pi / 180), math.sin((self.theta + 90) * math.pi / 180)],
            dtype='float64')

        self.pen.goto(float(thruster_pos1[0]), float(thruster_pos1[1]))
        self.pen.shape('square')
        self.pen.shapesize(1.5, 1)
        self.pen.setheading(self.theta - self.thrust1_angle)
        self.pen.color('Blue')
        self.pen.stamp()

        x2 = map(self.thrust2, 0, self.max_thrust + self.thrust_default, 3, 25)

        booster_pos2 = self.pos - 30 * np.array(
            [math.cos(self.theta * math.pi / 180), math.sin(self.theta * math.pi / 180)],
            dtype='float64') - x2 * np.array(
            [math.cos((self.theta + 90) * math.pi / 180), math.sin((self.theta + 90) * math.pi / 180)],
            dtype='float64')

        self.pen.goto(float(booster_pos2[0]), float(booster_pos2[1]))
        self.pen.shape('triangle')
        self.pen.shapesize(0.6, 1.5)
        self.pen.setheading(self.theta + self.thrust2_angle - 90)
        self.pen.color('orange')
        self.pen.stamp()

        thruster_pos2 = self.pos - 30 * np.array(
            [math.cos(self.theta * math.pi / 180), math.sin(self.theta * math.pi / 180)],
            dtype='float64') - 5 * np.array(
            [math.cos((self.theta + 90) * math.pi / 180), math.sin((self.theta + 90) * math.pi / 180)],
            dtype='float64')

        self.pen.goto(float(thruster_pos2[0]), float(thruster_pos2[1]))
        self.pen.shape('square')
        self.pen.shapesize(1.5, 1)
        self.pen.setheading(self.theta + self.thrust2_angle)
        self.pen.color('Blue')
        self.pen.stamp()

        self.pen.goto(float(self.target[0]), float(self.target[1]))
        self.pen.shape('circle')
        self.pen.shapesize(0.6, 0.6)
        self.pen.setheading(self.theta)
        self.pen.color('Red')
        self.pen.stamp()

        self.pen.color('white')

        self.pen.goto(-480, 360)
        self.pen.pendown()
        self.pen.write(f'Score: {self.score}',font=("Arial", 16, "normal"))
        self.pen.penup()

        self.pen.goto(-480, 340)
        self.pen.pendown()
        self.pen.write(f'Reward: {self.reward}',font=("Arial", 16, "normal"))
        self.pen.penup()

        self.pen.goto(-480, 320)
        self.pen.pendown()
        self.pen.write(f'Frames Left: {self.frames_left}',font=("Arial", 16, "normal"))
        self.pen.penup()

    def update_manual_run(self):
        net_f = (self.thrust1 + self.thrust2)
        x_acc = net_f * math.sin(-self.theta * math.pi / 180)
        y_acc = net_f * math.cos(-self.theta * math.pi / 180) - self.grav
        self.acc = np.array([x_acc, y_acc], dtype=np.float64) / self.mass
        self.ang_acc = (self.thrust1 - self.thrust2) * self.length / self.mass

        self.vel += self.acc
        self.vel *= self.air_friction
        self.pos += self.vel

        self.ang_vel += self.ang_acc
        self.ang_vel *= self.rot_friction
        self.theta += self.ang_vel

        self.frames_left -= 1

        if self.pos[0] > self.w / 2 or self.pos[0] < -self.w / 2:
            newp = np.array([self.pos[0] * -1, self.pos[1]])
            self.pos = newp

        if self.pos[1] > self.h / 2 or self.pos[1] < -self.h / 2:
            newp = np.array([self.pos[0], self.pos[1] * -1])
            self.pos = newp

        if self.check_target_collision():
            self.target_frame_ctr -= 1
        else:
            self.target_frame_ctr = self.target_frames

        if self.target_frame_ctr == 0:
            self.score += 1
            self.target_ctr += 1
            self.target_ctr %= len(self.targets)
            self.reward += self.frames_left / 50
            self.frames_left = self.frames_per_target
            if self.score < 12:
                self.target = self.targets[self.target_ctr]
            else:
                self.relocate()

        self.reward += 1 / (self.distance_to_target() + 1)
        if self.frames_left < 0:
            self.frames_left = 0

        self.ang_acc = 0
        self.acc *= 0
        self.thrust1 = self.thrust_default
        self.thrust2 = self.thrust_default

    def check_target_collision(self):
        r = self.pos - self.target
        dist = (r[0] ** 2 + r[1] ** 2) ** 0.5
        if dist < self.target_rad:
            return True
        else:
            return False

    def up(self):
        self.thrust1 += self.max_thrust
        self.thrust2 += self.max_thrust

    def down(self):
        self.thrust1 = 0
        self.thrust2 = 0

    def left(self):
        self.thrust1 += self.rot_default

    def right(self):
        self.thrust2 += self.rot_default

    def get_state(self):
        r = np.array(self.target, dtype=np.float64) - self.pos
        r /= 500
        inp1 = float(r[0])
        inp2 = float(r[1])
        inp3 = float(self.vel[0] / 10)
        inp4 = float(self.vel[1] / 10)
        inp5 = math.cos(self.theta)
        inp6 = math.sin(self.theta)
        inp7 = self.ang_vel / 5
        inp8 = 1

        return [inp1, inp2, inp3, inp4, inp5, inp6, inp7, inp8]

    def distance_to_target(self):
        r = self.pos - self.target
        dist = (r[0] ** 2 + r[1] ** 2) ** 0.5
        return dist

    def relocate(self):
        x = int(self.w / 2) - 100
        y = int(self.h / 2) - 100
        self.target = np.array([random.randint(-x, x), random.randint(-y, y)], dtype=np.float64)

    def relocate_to_mouse(self, x, y):
        self.target = np.array([x, y], dtype=np.float64)

    def run_manual(self):
        while True:
            self.wn.update()
            self.render()
            self.update_manual_run()
            time.sleep(0.02)

    def apply_action_from_nnOut(self, output):
        x1 = output[0]
        x2 = output[1]
        act1 = map(x1, 0, 1, 0, self.thrust_default + self.max_thrust)
        act2 = map(x2, 0, 1, 0, self.thrust_default + self.max_thrust)
        self.thrust1 += act1
        self.thrust2 += act2

    def update_nn_run(self):
        net_f = (self.thrust1 + self.thrust2)
        x_acc = net_f * math.sin(-self.theta * math.pi / 180)
        y_acc = net_f * math.cos(-self.theta * math.pi / 180) - self.grav
        self.acc = np.array([x_acc, y_acc], dtype=np.float64) / self.mass
        self.ang_acc = (self.thrust1 - self.thrust2) * self.length / self.mass

        self.vel += self.acc
        self.vel *= self.air_friction
        self.pos += self.vel

        self.ang_vel += self.ang_acc
        self.ang_vel *= self.rot_friction
        self.theta += self.ang_vel

        if self.pos[0] > (self.w + 200) / 2 or self.pos[0] < -(self.w + 200) / 2:
            self.running = False

        if self.pos[1] > (self.h + 200) / 2 or self.pos[1] < -(self.h + 200) / 2:
            self.running = False

        self.frames_left -= 1

        if self.check_target_collision():
            self.target_frame_ctr -= 1
        else:
            self.target_frame_ctr = self.target_frames

        if self.target_frame_ctr == 0:
            self.score += 1
            self.target_ctr += 1
            self.target_ctr = self.target_ctr%(len(self.targets))
            self.reward += self.frames_left/20
            self.frames_left = self.frames_per_target
            x = int(self.w / 2) - 10
            y = int(self.h / 2) - 10
            self.target = np.array([random.randint(-x, x), random.randint(-y, y)], dtype=np.float64)
            self.reward += 200
            '''
            if self.score < 12:
                self.target = self.targets[self.target_ctr]
            else:
                x = int(self.w / 2) - 10
                y = int(self.h / 2) - 10
                self.target = np.array([random.randint(-x, x), random.randint(-y, y)], dtype=np.float64)
                self.reward += 200
                self.running = False'''

        if self.frames_left == 0:
            self.running = False

        #self.reward += 1/(self.distance_to_target() + 1)

        self.ang_acc = 0
        self.acc *= 0
        self.thrust1 = self.thrust_default
        self.thrust2 = self.thrust_default

    def reset(self):
        self.pos = np.array([0, 0], dtype=np.float64)
        self.vel = np.array([0, 0], dtype=np.float64)
        self.acc = np.array([0, 0], dtype=np.float64)

        self.theta = 0
        self.ang_vel = 0
        self.ang_acc = 0

        self.score = 0
        self.reward = 0
        self.target = self.targets[self.seed]
        self.target_frame_ctr = self.target_frames
        self.running = True
        self.frames_left = self.frames_per_target


    def run_ind(self,ind):
        while self.running:
            self.update_nn_run()
            inp = self.get_state()
            out = ind.brain.forward(inp)
            self.apply_action_from_nnOut(out)
            self.frames_left += 1

            if self.display:
                self.render()
                self.wn.update()
                time.sleep(0.02)
        self.reset()

    def evaluate_ind(self,ind):
        while self.running:
            self.update_nn_run()
            inp = self.get_state()
            out = ind.brain.forward(inp)
            self.apply_action_from_nnOut(out)

            if self.display:
                print(1)
                self.render()
                self.wn.update()
                time.sleep(0.02)
        ind.fitness = self.reward
        self.reset()

    def next_gen(self):
        self.gen_n += 1
        x = int(self.w / 2) - 10
        y = int(self.h / 2) - 10
        targets = [np.array([-450,380])]
        for i in range(len(self.targets)-1):
            xcor = random.randint(x-150, x) * random.choice([-1,1])
            ycor = random.randint(y-150, y) * random.choice([-1,1])
            new_tar = np.array([xcor,ycor], dtype=np.float64)
            r = new_tar - targets[-1]
            dist = (r[0] ** 2 + r[1] ** 2) ** 0.5
            while dist < 400:
                xcor = random.randint(x - 150, x) * random.choice([-1, 1])
                ycor = random.randint(y - 150, y) * random.choice([-1, 1])
                new_tar = np.array([xcor, ycor], dtype=np.float64)
                r = new_tar - targets[-1]
                dist = (r[0] ** 2 + r[1] ** 2) ** 0.5
            targets.append(new_tar)

        rand = random.choices([i for i in range(len(self.targets))],k = 3)
        for i in rand:
            targets[i] = np.array([random.randint(-x, x), random.randint(-y, y)], dtype=np.float64)
        if self.gen_n % 20 == 0:
            targets = []
            for i in range(len(self.targets)):
                x = int(self.w / 2) - 100
                y = int(self.h / 2) - 100
                targets.append(np.array([random.randint(-x, x), random.randint(-y, y)], dtype=np.float64))
        self.targets = targets

if __name__ == '__main__':
    x = DroneSim(manual=True,display=True)
    x.run_manual()
    for i in range(100):
        ind = Individual()
        x.run_ind(ind)


