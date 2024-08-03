from stim_behavior.utils.logger import Logger
from stim_behavior.utils.utils import load_config

class ManagerBase:
    def __init__(self, log_level="INFO"):
        self.logger = Logger(log_level=log_level)

        self.path = load_config(config_file='tshrew_paths_user')
        self.params = load_config(config_file='tshrew_params', override_file='tshrew_params.override')
        self.logger.debug(f"Path: {self.path}")
        self.logger.info(f"Params: {self.params}")

    def _setattr(self, key, value):
        setattr(self, key, value)
        self.logger.debug(f"[{key}] added to the object")

    def _set_new_path(self, path_key, path_value):
        self.path[path_key] = path_value
        self.logger.debug(f"[{path_key}] path added")

    def load_session(self, session_id, subsession_id):
        self._setattr('session_id', session_id)
        self._setattr('subsession_id', subsession_id)

        subpath = f"Session_{session_id}/{subsession_id}"
        self._set_new_path('session_subpath', subpath)    
        self.logger.debug(f"Session [{session_id}], subsession [{subsession_id}] loaded")