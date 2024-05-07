from models import CustomModule


class TrainingScheduler(CustomModule):

    def __init__(self, name):
        super().__init__(name)

    def should_train(self, name):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def get_current_level(self):
        raise NotImplementedError()


class SimultaneousTrainingScheduler(TrainingScheduler):

    def __init__(self):
        super().__init__('SimultaneousTrainingScheduler')

    def should_train(self, name) -> bool:
        return True

    def step(self) -> bool:
        return False

    def get_current_level(self) -> str:
        return 'simultaneous'


class LevelTrainingScheduler(TrainingScheduler):

    def __init__(self, names: list, steps: list):
        super().__init__('LevelTrainingScheduler')

        assert len(names) == len(
            steps), f'Length of names and steps should be the same. Got {len(names)} and {len(steps)}.'

        self.iteration = 0
        self.global_step = 0
        self.steps = steps
        self.names = names
        self.counts = len(names)

    def should_train(self, name):
        if name not in self.names:
            return False
        return self.iteration % self.counts == self.names.index(name)

    def get_current_level(self):
        return self.names[self.iteration % self.counts]

    def step(self):
        """
            This function steps the scheduler.
        :return: True if the iteration is finished
        """

        self.global_step += 1

        if self.global_step >= self.steps[self.iteration % self.counts]:
            self.iteration += 1
            self.global_step = 0
            return True

        return False
