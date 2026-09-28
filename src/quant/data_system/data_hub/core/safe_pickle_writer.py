import os
import uuid
import pickle

class SafePickleWriter:
    @staticmethod
    def safe_to_pickle(obj,path:str)->None:
        dir_path = os.path.dirname(path) or "."
        os.makedirs(dir_path,exist_ok=True)
        tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"

        try:
            with open(tmp_path,"wb") as f:
                pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path,path)

            dir_fd = os.open(dir_path,os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise