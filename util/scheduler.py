from models import CustomModule


class LevelTrainingScheduler(CustomModule):

    def __init__(self, names, steps):
        super().__init__('LevelTrainingScheduler')

        self.iteration = 0
        self.global_step = 0
        self.steps = steps
        self.names = names
        self.counts = len(names)

    def should_train(self, name):
        return self.iteration % self.counts == self.names.index(name)

    def get_current_level(self):
        return self.names[self.iteration % self.counts]

    def step(self):
        """
            This function steps the scheduler.
        :return: True if the iteration is finished
        """

        self.global_step += 1

        if self.global_step >= self.steps:
            self.iteration += 1
            self.global_step = 0
            return True

        return False
