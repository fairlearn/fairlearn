#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:53:18 2019

@author: SRAYAGARWAL
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Mean difference function
def mean_diff(data, protected_attribute_name, label, privileged_group_protected_attribute_value,unprivileged_group_protected_attribute_value):
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

    #x1.plot(kind='bar', title="Mean of target of two groups of the protected class")
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

    # x1 = pd.Series(priv_pos_length,unpriv_pos_length)
    # x1.index = ['Priv','UnPriv']
    # x1.plot(kind='bar', title="xxxx two groups of the protected class")

    return priv_pos_length, unpriv_pos_length, difference, ratio


def neg_class_diff(data, protected_attribute_name, label, privileged_group_protected_attribute_value,unprivileged_group_protected_attribute_value):
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


def stat_parity_test(data, protected_attribute_name, label, fav, privileged_group_protected_attribute_value,unprivileged_group_protected_attribute_value):
    priv_subset = data[data[protected_attribute_name] == privileged_group_protected_attribute_value]
    priv_count_pos = len(priv_subset[priv_subset[label] == fav])

    unpriv_subset = data[data[protected_attribute_name] == unprivileged_group_protected_attribute_value]
    unpriv_count_pos = len(unpriv_subset[unpriv_subset[label] == fav])

    total_priv = len(priv_subset)
    total_unpriv = len(unpriv_subset)

    stat_parity = (unpriv_count_pos / total_unpriv) - (priv_count_pos / total_priv)
    disprate_impact = (unpriv_count_pos / total_unpriv) / (priv_count_pos / total_priv)

    print ("Statistical Parity Difference", stat_parity)
    print ("Disparate Impact", disprate_impact)

    return stat_parity, disprate_impact


def missing_value(data, protected_attribute_name, label, privileged_group_protected_attribute_value,unprivileged_group_protected_attribute_value):
    priv_subset = data[data[protected_attribute_name] == privileged_group_protected_attribute_value]
    # priv_missing = priv_subset.isna().sum()
    tot_priv_missing = priv_subset.isna().sum().sum()

    unpriv_subset = data[data[protected_attribute_name] == unprivileged_group_protected_attribute_value]
    # unpriv_missing = unpriv_subset.isna().sum()
    tot_unpriv_missing = unpriv_subset.isna().sum().sum()

    diff_missing = tot_unpriv_missing - tot_priv_missing
    ratio_missing = (tot_unpriv_missing+0.001) / (tot_priv_missing+0.001)

    priv_subset_label_1 = priv_subset[priv_subset[label] == 1]
    unpriv_subset_label_1 = unpriv_subset[unpriv_subset[label] == 1]

    # priv_missing_1 = priv_subset_label_1.isna().sum()
    tot_priv_missing_1 = priv_subset_label_1.isna().sum().sum()

    # unpriv_missing_1 = unpriv_subset_label_1.isna().sum()
    tot_unpriv_missing_1 = unpriv_subset_label_1.isna().sum().sum()

    diff_label_1 = tot_unpriv_missing_1 - tot_priv_missing_1
    ratio_label_1 = (tot_unpriv_missing_1 + 0.001) / (tot_priv_missing_1 +0.001)

    print ("Missing value priviliged", tot_priv_missing)
    print ("Missing value unpriviliged", tot_unpriv_missing)
    print ("Missing value difference", diff_missing)
    print ("Missing value ratio", ratio_missing)
    print ("Missing value difference for favourable outcome", diff_label_1)
    print ("Missing value ratio for favourable outcome", ratio_label_1)

    # return priv_missing, unpriv_missing, diff_missing, ratio_missing, diff_label_1, ratio_label_1

    fig = plt.figure(figsize=(10, 10))  # width x height
    ax1 = fig.add_subplot(3, 2, 1)  # row, column, position
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)

    ax1.set_title('Priviliged')
    ax2.set_title('UnPriviliged')
    ax3.set_title('Priviliged & Favourable')
    ax4.set_title('UnPriviliged & Favourable')

    # plt.tight_layout()
    plt.tight_layout(pad=.2, w_pad=.5, h_pad=12.0)

    sns.heatmap(priv_subset.isnull(), ax=ax1, cbar=False, yticklabels=False)
    sns.heatmap(unpriv_subset.isnull(), ax=ax2, cbar=False, yticklabels=False)
    sns.heatmap(priv_subset_label_1.isnull(), ax=ax3, cbar=False, yticklabels=False)
    sns.heatmap(unpriv_subset_label_1.isnull(), ax=ax4, cbar=False, yticklabels=False)

    return tot_priv_missing, tot_unpriv_missing, diff_missing, ratio_missing, diff_label_1, ratio_label_1

