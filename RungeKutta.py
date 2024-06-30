import numpy as np

class RK4:

  def __init__(self, initial_conditions, step_function, step):
    self.initial_conditions = initial_conditions
    self.func = step_function
    self.step = np.float128(step)

    self.state = np.array(initial_conditions, dtype=np.float128)
    self.time = np.float128()

  def next(self):
    k1 = self.step * self.func(self.time, np.array(self.state))
    k2 = self.step * self.func(self.time + self.step / 2, np.array(self.state) + k1 / 2)
    k3 = self.step * self.func(self.time + self.step / 2,np.array(self.state) + k2 / 2)
    k4 = self.step * self.func(self.time + self.step, np.array(self.state) + k3)
  
    increment = (k1 + 2 * k2 + 2 * k3 + k4) / 6

    self.state += increment
    self.time += self.step

    return (self.time, increment, self.state)

  def reset(self):
    self.state = np.array(self.initial_conditions, dtype=np.float128)
    self.time = np.float128()

  def solve_until(self, final_time):
    self.reset()

    state_output = self.initial_conditions[:, :, np.newaxis]
    t = [0]

    last_time = t[0]
    while True:
      time, increment, state = self.next()

      t.append(time)
      state_output = np.concatenate((state_output, state[:, :, np.newaxis]), axis=2)
      
      last_time = time
      if last_time > final_time + self.step:
        break

    return (t,state_output)
  