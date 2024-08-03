import os
import sys
import suite2p
from .manager import ManagerBase
from stim_behavior.utils.logger import Logger
from stim_behavior.utils.utils import *


class Suite2pManager(ManagerBase):
    def __init__(self, log_level="INFO"):
        super().__init__(log_level=log_level)
        self.output_keys = ["F", "Fneu", "spks", "iscell", "ops", "stat"]

    def _setattr(self, *args, **kwargs):
        return super()._setattr(*args, **kwargs)

    def _set_new_path(self, *args, **kwargs):
        return super()._set_new_path(*args, **kwargs)

    def load_session(self, *args, **kwargs):
        return super().load_session(*args, **kwargs)
    
    def load_neural_session(self, session_id, subsession_id):
        self.load_session(session_id, subsession_id)
        subpath = f"Session_{session_id}/{subsession_id}"
        self._set_new_path('suite2p', f"{self.path['processed_data']}/{subpath}/suite2p/plane0")
        
    def _is_suite2p_processed(self):
        try:
            files = os.listdir(self.path['suite2p'])
            files = [file.split('.')[0] for file in files]
            for s2p_out in self.output_keys:
                if s2p_out not in files:
                    self.logger.debug(f"[{s2p_out}] is not processed. Suite2p will run.")
                    return False
            self.logger.debug("All suite2p files processed")
            return True
        except FileNotFoundError:
            self.logger.debug("Suite2p file/folder missing. Suite2p will run.")
            return False
    
    def run(self, force_run = False):
        if "session_subpath" not in self.path:
            self.logger.error("Session not loaded. Call load_session() first")
            return
        
        if self._is_suite2p_processed() and not force_run:
            self.logger.info(f"Suite2p already processed for current session. Skipping...")
            return
        
        subpath = self.path['session_subpath']
        ops = np.load(f"{self.path['raw_data']}/ops.npy", allow_pickle=True).tolist()
        ops["save_path0"] = f"{self.path['processed_data']}/{subpath}"
        db = {
            'data_path':[f"{self.path['raw_data']}/{subpath}"]
        }
        suite2p.run_s2p(ops=ops, db=db)        
        # Delete data.bin since it's a huge and unneeded file
        try:
            os.remove(f"{self.path['suite2p']}/data.bin")
        except:
            self.logger.warning(f"Failed to delete data.bin")
        self.logger.info(f"Suite2p neural traces saved!")

    def compute_dF(self, save_data=True):
        """Extracting dF from F and Fneu"""
        if not self._is_suite2p_processed():
            self.logger.warn(f"Run Suite2p first before calling this. Skipping...")
            return
        
        F = np.load(f'{self.path["suite2p"]}/F.npy')
        Fneu = np.load(f'{self.path["suite2p"]}/Fneu.npy')
        ops = np.load(f'{self.path["suite2p"]}/ops.npy', allow_pickle=True).item()
        dF = F.copy() - 0.7*Fneu
        dF = suite2p.extraction.preprocess(dF, ops['baseline'], ops['win_baseline'], 
                                   ops['sig_baseline'], ops['fs'], ops['prctile_baseline'])
        self._setattr("dF", dF)
        if save_data:
            np.save(f"{self.path['suite2p']}/dF.npy", dF)
            self.logger.info(f"dF computed and saved at {self.path['suite2p']}/dF.npy")
        self.logger.info(f"dF: {dF.shape}")
        return dF
    
    def load_traces(self):
        obj_names = ['F', 'Fneu', 'spks', 'iscell']
        suite2p_path = self.path['suite2p']

        data = {}

        for obj_name in obj_names:
            obj = np.load(f'{suite2p_path}/{obj_name}.npy')
            data[obj_name] = obj
            self.logger.info(f"{obj_name}: {obj.shape}")

        ops = np.load(f'{suite2p_path}/ops.npy', allow_pickle=True).item()
        data["ops"] = ops
        stat = np.load(f'{suite2p_path}/stat.npy', allow_pickle=True)
        self._setattr("stat", stat)
        data["stat"] = stat
        data["ROI_COUNT"] = data['F'].shape[0]
        data["FRAME_COUNT"] = data['F'].shape[1]

        return data