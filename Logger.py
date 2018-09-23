#!/user/bin/env python
# -*- coding:utf-8 -*-
import logging
import config

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s',
                    filename=config.log_path, filemode='a')
logger = logging.getLogger(__name__)
fh = logging.FileHandler(config.log_path)
sh = logging.StreamHandler()

# 定义handler的输出格式formatter
formatter = logging.Formatter('%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)
logger.setLevel(logging.DEBUG)
