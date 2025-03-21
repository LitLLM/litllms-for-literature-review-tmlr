#!/usr/bin/env python3
# Copyright (c) ServiceNow Research and its affiliates.
"""
This file defines base class to be inherited by different models
"""
import abc


class BaseMLModel(abc.ABC):
    """
    Base class inherited by all the ML models
    """

    @abc.abstractmethod
    def gen_response(self, **kwargs):
        """
        Each class should implement this function
        """
        pass
