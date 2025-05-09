import copy
from abc import ABC, abstractmethod


class FaultModeGenerator(ABC):
    @abstractmethod
    def generate_fault_mode_function(self, args):
        pass


class FaultModeGeneratorDiscrete(FaultModeGenerator):
    def generate_fault_mode_function(self, args):
        mapping = eval(args)

        def fault_mode(a):
            return mapping[a]

        return fault_mode
