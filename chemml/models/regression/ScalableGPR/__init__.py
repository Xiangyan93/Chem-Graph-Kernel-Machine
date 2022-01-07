#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .nystrom import LRAGPR
from .NLE import NaiveLocalExpertGP as NLEGPR


__all__ = [
    'NLEGPR', 'LRAGPR'
]
