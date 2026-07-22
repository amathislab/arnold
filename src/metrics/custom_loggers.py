from imitation.algorithms.bc import BCLogger


class WandbBCLogger(BCLogger):

    def __init__(self, logger):
        self._logger = logger
        self._temp_dict = {}

    def record(self, name: str, val: float):
        cur_dict = self._temp_dict
        splited_name = name.split("/")
        for ks in splited_name[:-1]:
            if ks not in cur_dict:
                cur_dict[ks] = {}
            cur_dict = cur_dict[ks]
        cur_dict[splited_name[-1]] = val

    def dump(self, step: int):
        self._logger.log(self._temp_dict, step=step, commit=True)
