import numpy as np
from typing import Union

from ceyehao.utils.data_process import tt_convert, p_transform, obs_params2imgs
from ceyehao.tools.infer import TTPredictor


class FlowSystem(object):
    def __init__(self, res: int = 200, num_obs=8, profile_channel=3):
        self.__profiles = np.zeros(
            [num_obs + 1, res, res, profile_channel]
        )  # inlcuding input and intermediate
        self.__obs_coord = np.zeros([num_obs, 19])  # (x0, x1, y) * 6 + pos
        self.__tts = np.zeros([num_obs, res, res, 2]) 
        self.__updated = {"tt": False, "p": False}

    def set_pin(self, pin):
        self.__profiles[0] = pin
        self.__updated["p"] = False

    def set_obs(self, obs: Union[list, np.ndarray]):
        if type(obs) == list:
            param, pos = obs
        elif type(obs) == np.ndarray:
            param = obs[:, :-1]
            pos = obs[:, -1]
        self.__obs_coord = param
        self.__obs_pos = pos
        self.__updated["tt"] = False
        self.__updated["p"] = False

    def update_tts(self, tt_predictor: TTPredictor):
        obs_imgs = self.obs_imgs()
        tts = tt_predictor.predict_from_obs_img(obs_imgs)
        self.__tts = tt_convert(tts, vert_sym=self.__obs_pos)
        self.__updated["tt"] = True

    def update_p(self):
        for i in range(self.__profiles.shape[0]):
            if i == 0:
                continue
            self.__profiles[i] = p_transform(self.__profiles[i - 1], self.__tts[i - 1])
        self.__updated["p"] = True

    def update(self, tt_predictor: TTPredictor):
        self.update_tts(tt_predictor)
        self.update_p()

    def p_tensors(self, check_status=True):
        if check_status:
            assert self.__updated["p"], "profiles are not updated yet."
        return self.__profiles.copy()

    def p_imgs(self, check_status=True):
        if check_status:
            assert self.__updated["p"], "profiles are not updated yet."
        return self.__profiles[:, :, ::-1, :].transpose([1, 0, 2])

    def obs_imgs(self):
        return obs_params2imgs(self.__obs_coord, self.__obs_pos)

    def tts(self, check_status=True):
        if check_status:
            assert self.__updated["tt"], "tt are not updated yet."
        return self.__tts.copy()

    def status(self):
        return self.__updated
