#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: heatmaps.py
Description: Generates heatmap data based on trial types provided. Inherits classes from coefficient_cluster because these functions and
    objects organize the data already. 
Author: David Estrin
Version: 1.0
Date: 12-06-2024

Note: Portions of code are based on code from Drs. Puja Parekh & Jesse Keminsky, Parekh et al., 2024 
"""

from Sweet2Plus.statistics.coefficient_clustering import regression_coeffecient_pca_clustering, gather_data