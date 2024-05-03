import signal
import torch
import subprocess
import numpy as np
import redis
import ast


class DB():
    def __init__(self):
        # self.db_server = subprocess.Popen(["/usr/bin/redis-server",], close_fds=True)
        self.client = redis.Redis(host="localhost", port=6379, decode_responses=True)             

    def stop(self):
        self.client.shutdown()
        if self.db_server.poll() is None:
            self.db_server.terminate()
            self.db_server.wait()
    
    def get(self, key):
        res = self.client.get(key)
        emb = torch.tensor(ast.literal_eval(res))
        # emb = torch.from_numpy(np.frombuffer(res, dtype=np.float32))
        return emb
    
    def put(self, key, val):
        val = val.cpu()
        # val = val.numpy().tobytes()
        val = str(val.tolist())
        res = self.client.set(key,val)

        

if __name__ == "__main__":
    # client = redis.Redis(host="localhost", port=6379, decode_responses=False)
    db = DB()
    # db.put(1, torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.float32))
    res = db.get(7)
    print(res)
    print(res.shape)


