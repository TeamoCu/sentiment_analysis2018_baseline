#!/user/bin/env python
# -*- coding:utf-8 -*-
import logging
import config

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s',
                    filename=config.log_path, filemode='a')
logger = logging.getLogger(__name__)
sh = logging.StreamHandler()

# 定义handler的输出格式formatter
formatter = logging.Formatter('%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel(logging.DEBUG)
