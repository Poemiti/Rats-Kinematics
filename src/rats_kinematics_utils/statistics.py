#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
from statsmodels.stats.multitest import multipletests




def ANOVA(data: pd.DataFrame, formula: str = "velocity ~ condition * laser_state * laser_intensity"):

    from rats_kinematics_utils.plot_comparative import _trim_extremes_iqr

    # data_trimmed = _trim_extremes_iqr(data, k=1.5).copy()
    data_trimmed = data

    for col in ["condition", "laser_state", "laser_intensity"]:
        data_trimmed[col] = data_trimmed[col].astype("category")

    # Fit model
    model = smf.ols(formula, data=data_trimmed).fit()
    anova_results = anova_lm(model, typ=2)

    print("\nANOVA RESULTS")
    print(anova_results)

    # post hoc if p-value is significant
    for effect in ["condition", "laser_state", "laser_intensity"]:
        pval = anova_results.loc[effect, "PR(>F)"]

        if pval < 0.05:
            print(f"\nPost-hoc for {effect} (p = {pval:.4f})")

            tukey = pairwise_tukeyhsd(
                endog=data_trimmed["velocity"],
                groups=data_trimmed[effect],
                alpha=0.05
            )

            print(tukey)

        else : 
            print(f"\nNot significant: {effect} (p = {pval:.4f})")




def kruskal_test(group):
    # create one array per condition
    samples = [
        sub_group["value"].values
        for _, sub_group in group.groupby("condition")
    ]
    
    # need at least 2 groups
    if len(samples) < 2:
        return pd.Series({
            "n_groups": len(samples),
            "kruskal_stat": None,
            "p_value": None
        })
    
    stat, p = stats.kruskal(*samples)
    
    return pd.Series({
        "n_groups": len(samples),
        "kruskal_stat": stat,
        "p_value": p
    })

def shapiro_test(group):
    stat, p = stats.shapiro(group["value"])
    return pd.Series({
        "n": len(group),
        "shapiro_stat": stat,
        "p_value": p
    })



def compute_statistics(data: pd.DataFrame, formula: str) : 

    # 1. check normality
    print("\n1. Checking normality: ")

    data_normality = (data
                          .groupby(["condition", "laser_intensity"])
                          .apply(shapiro_test, include_groups=False)
                          .reset_index()
                    )


    print("\nNORMAL: ")
    print(data_normality[data_normality["p_value"] > 0.05])
    print("\nNOT NORMAL: ")
    print(data_normality[data_normality["p_value"] < 0.05])

    # 2. Kruskal test
    print("\n2. Making Kruskal Wallis test: ")
    data_kruskal = kruskal_test(data)
    print("SIGNIFICANT" if data_kruskal["p_value"] < 0.05 else "NOT SIGNIFICANT")
    print(data_kruskal)

    data["group"] = (data["condition"] + "." + data["laser_intensity"])

    comparisons = [
                    # Conti vs Beta
                    ("Conti_LaserOff.low",  "Beta_LaserOff.low"),
                    ("Conti_LaserOff.high", "Beta_LaserOff.high"),
                    ("Conti_LaserOn.low",   "Beta_LaserOn.low"),
                    ("Conti_LaserOn.high",  "Beta_LaserOn.high"),

                    # Off vs On
                    ("Conti_LaserOff.low",  "Conti_LaserOn.low"),
                    ("Conti_LaserOff.high", "Conti_LaserOn.high"),
                    ("Beta_LaserOff.low",   "Beta_LaserOn.low"),
                    ("Beta_LaserOff.high",  "Beta_LaserOn.high"),

                    # low vs high
                    ("Beta_LaserOff.low",   "Beta_LaserOff.high"),
                    ("Beta_LaserOn.low",    "Beta_LaserOn.high"),
                    ("Conti_LaserOff.low",  "Conti_LaserOff.high"),
                    ("Conti_LaserOn.low",   "Conti_LaserOn.high"),
                ]

    results = []

    print(data)
    for g1, g2 in comparisons:
        x = data.loc[data["group"] == g1, "value"]
        y = data.loc[data["group"] == g2, "value"]
        
        if len(x) > 0 and len(y) > 0:
            stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        else:
            stat, p = None, None
        
        results.append({
            "group1": g1,
            "group2": g2,
            "n1": len(x),
            "n2": len(y),
            "U_stat": stat,
            "p_value": p
        })


    print("\n3. Mann Whitney pairwise comparaison:")
    pairwise_results = pd.DataFrame(results)
    pairwise_results["p_adj"] = multipletests(pairwise_results["p_value"], method="fdr_bh")[1]
    
    print(pairwise_results)
    print("\nSIGNIFICANT: ")
    print(pairwise_results[pairwise_results["p_value"] < 0.05])
    print("\nNOT SIGNIFICANT: ")
    print(pairwise_results[pairwise_results["p_value"] > 0.05])

    return pairwise_results[pairwise_results["p_value"] < 0.05]







