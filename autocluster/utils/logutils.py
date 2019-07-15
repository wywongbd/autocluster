from datetime import datetime
from pytz import timezone

import os
import pathlib
import logging

class LogUtils(object):
    @staticmethod
    def create_new_directory(prefix):
        folder_name = 'log/{}-'.format(prefix)
        folder_name += '{}'
        output_dir = folder_name.format(datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-1])
        
        if not os.path.exists(output_dir):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        return output_dir