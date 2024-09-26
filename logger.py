import os
import time


class Logger(object):
    def __init__(self, log_dir, log_name):
        self.log_dir = log_dir
        self.log_name = log_name

        # 更新 日志文件路径
        self.log_dir = os.path.join(self.log_dir, self.log_name)
        # 检查目录是否存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # 创建日志文件
        self.log_path = os.path.join(self.log_dir, "log.txt")
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.close()

    # 写入日志
    def write_log(self, log: str, mode: str,  info="/"):
        head = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}, mode={mode.upper()}, info={info}]" 
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{head}{log}\n")
            f.close()
