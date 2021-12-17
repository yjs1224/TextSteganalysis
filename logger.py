# -*- coding: utf-8 -*-

import logging


class Logger(object):
    def __init__(self, log_file):
        self.logger    = logging.getLogger()
        self.formatter = logging.Formatter(fmt='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(self.formatter)
        self.logger.addHandler(sh)

    def info(self, text):
        self.logger.info(text)
