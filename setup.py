#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:31:13 2021

@author: ying
"""


from setuptools import setup

setup(name='bsr',
      version='0.1',
      description='A Bayesian MCMC based Symbolic Regression Algorithm',
      author='Ying Jin',
      author_email='yjin1827@gmail.com',
      url='https://github.com/ying531/MCMC-SymReg',
      packages = ['MCMC-SymReg'],
      package_dir = {'MCMC-SymReg':'codes'}
)