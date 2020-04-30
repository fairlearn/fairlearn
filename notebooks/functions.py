#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:53:18 2019

@author: SRAYAGARWAL
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Mean difference function
def mean_diff(data, protected_attribute_name, label, privileged_group_protected_attribute_value, unprivileged_group_protected_attribute_value):
    subset_priv = data[data[protected_attribute_name] == privileged_group_protected_attribute_value]
    subset_unpriv = data[data[protected_attribute_name] == unprivileged_group_protected_attribute_value]
    print('mean of target for privilged group', np.mean(subset_priv[label]))
    print('mean of target for privilged group', np.mean(subset_unpriv[label]))
    mean = np.mean(subset_priv[label]) - np.mean(subset_unpriv[label])
    print('mean difference of target for protected class', mean)
    mean_ratio = np.mean(subset_priv[label]) / np.mean(subset_unpriv[label])
    print('mean ratio of target for protected class', mean_ratio)
    x1 = pd.Series([np.mean(subset_priv[label]), np.mean(subset_unpriv[label])])
    x1.index = ['Priv', 'UnPriv']
    plt.bar(x1.index, x1)
    plt.savefig('foo.png')
    return mean, mean_ratio


def pos_class_diff(data, protected_attribute_name, label, privileged_group_protected_attribute_value, unprivileged_group_protected_attribute_value):
    priv_count_subset = data[data[protected_attribute_name] == privileged_group_protected_attribute_value]
    priv_count = priv_count_subset[priv_count_subset[label] == 1]
    priv_count2 = priv_count[label]
    priv_pos_length = len(priv_count2)
    print("len of positive labrl for priviliged group", priv_pos_length)
    unpriv_count_subset = data[data[protected_attribute_name] == unprivileged_group_protected_attribute_value]
    unpriv_count = unpriv_count_subset[unpriv_count_subset[label] == 1]
    unpriv_count2 = unpriv_count[label]
    unpriv_pos_length = len(unpriv_count2)
    print("len of positive labrl for unpriviliged group", unpriv_pos_length)
    difference = priv_pos_length - unpriv_pos_length
    ratio = priv_pos_length / unpriv_pos_length
    print("diff len of positive label", difference)
    print("ratio len of positive label", ratio)
    return priv_pos_length, unpriv_pos_length, difference, ratio


def neg_class_diff(data, protected_attribute_name, label, privileged_group_protected_attribute_value, unprivileged_group_protected_attribute_value):
    priv_count_subset = data[data[protected_attribute_name] == privileged_group_protected_attribute_value]
    priv_count = priv_count_subset[priv_count_subset[label] == 0]
    priv_count2 = priv_count[label]
    priv_neg_length = len(priv_count2)
    print("len of negative labrl for priviliged group", priv_neg_length)
    unpriv_count_subset = data[data[protected_attribute_name] == unprivileged_group_protected_attribute_value]
    unpriv_count = unpriv_count_subset[unpriv_count_subset[label] == 0]
    unpriv_count2 = unpriv_count[label]
    unpriv_neg_length = len(unpriv_count2)
    print("len of negative labrl for unpriviliged group", unpriv_neg_length)
    neg_difference = priv_neg_length - unpriv_neg_length
    print("diff len of negative label", neg_difference)
    neg_ratio = priv_neg_length / unpriv_neg_length
    print("ratio len of negative label", neg_ratio)
    return priv_neg_length, unpriv_neg_length, neg_difference, neg_ratio


def pos_neg_class_diff(data, protected_attribute_name, label, privileged_group_protected_attribute_value, unprivileged_group_protected_attribute_value):
    priv_pos_length, unpriv_pos_length, difference, ratio = pos_class_diff(data, protected_attribute_name,
                                                                           label, privileged_group_protected_attribute_value, unprivileged_group_protected_attribute_value)
    priv_neg_length, unpriv_neg_length, neg_difference, neg_ratio = neg_class_diff(data, protected_attribute_name,
                                                                                   label, privileged_group_protected_attribute_value, unprivileged_group_protected_attribute_value)

    privD = priv_pos_length - priv_neg_length
    print("---", privD)
    unprivD = unpriv_pos_length - unpriv_neg_length
    print("====", unprivD)
    FullD = privD - unprivD
    FullR = privD / unprivD
    return FullD, FullR


def stat_parity_test(data, protected_attribute_name, label, fav, privileged_group_protected_attribute_value, unprivileged_group_protected_attribute_value):
    priv_subset = data[data[protected_attribute_name] == privileged_group_protected_attribute_value]
    priv_count_pos = len(priv_subset[priv_subset[label] == fav])
    unpriv_subset = data[data[protected_attribute_name] == unprivileged_group_protected_attribute_value]
    unpriv_count_pos = len(unpriv_subset[unpriv_subset[label] == fav])
    total_priv = len(priv_subset)
    total_unpriv = len(unpriv_subset)
    stat_parity = (unpriv_count_pos / total_unpriv) - (priv_count_pos / total_priv)
    disprate_impact = (unpriv_count_pos / total_unpriv) / (priv_count_pos / total_priv)
    print("Statistical Parity Difference", stat_parity)
    print("Disparate Impact", disprate_impact)
    return stat_parity, disprate_impact
