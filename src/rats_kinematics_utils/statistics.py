#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from itertools import permutations
import pingouin as pg

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


def mann_whitney(data: pd.DataFrame, comparisons: list) : 
    results = []

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


    pairwise_results = pd.DataFrame(results)
    pairwise_results["p_adj"] = multipletests(pairwise_results["p_value"], method="fdr_bh")[1]
    
    return pairwise_results


def compute_statistics(data: pd.DataFrame, formula: str) : 
    stat_res = {}

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

    stat_res["shapiro"] = data_normality

    # 2. Kruskal test
    print("\n2. Making Kruskal Wallis test: ")
    data_kruskal = kruskal_test(data)
    print(data_kruskal)

    stat_res["kruskal"] = data_kruskal

    if data_kruskal["p_value"] > 0.05 : 
        print(f"\nNOT SIGNIFICANT")
        return stat_res

    print("\nSIGNIFICANT")

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

    print("\n3. Mann Whitney pairwise comparaison:")
    pairwise_results = mann_whitney(data, comparisons)
    
    print(pairwise_results)
    print("\nSIGNIFICANT: ")
    print(pairwise_results[pairwise_results["p_value"] < 0.05])
    print("\nNOT SIGNIFICANT: ")
    print(pairwise_results[pairwise_results["p_value"] > 0.05])

    stat_res["mann_whitney"] = pairwise_results

    return stat_res


def save_stat_results(data: dict, path: Path) : 
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, path)



def LMM(data, formula): 
    data["rat"] = data["rat"].astype("category")

    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["rat"],
        # re_formula="~condition"
    )

    result = model.fit(method="lbfgs")
    residuals = model.resid

    plt.hist(residuals, bins=30)
    plt.show()

    s, p_val = stats.shapiro(residuals)
    print(f"Shapiro on residues : {p_val}")

    return result











############ permutation ##################





def permutation(group1, group2, n_perm=10000):
    values1 = group1["value"].to_numpy()
    values2 = group2["value"].to_numpy()
    
    n1 = len(values1)
    n2 = len(values2)
    print(n1, n2)
    combined = np.concatenate([values1, values2])
    
    # observed difference
    observed_diff = np.abs(values1.mean() - values2.mean())
    
    perm_diff = np.zeros(n_perm)
    for i in range(n_perm):
        shuffled = np.random.permutation(combined)
        perm_diff[i] = np.abs(shuffled[:n1].mean() - shuffled[n1:n1+n2].mean())
    
    # two-tailed p-value
    p_value = np.mean(perm_diff >= observed_diff)
    
    return observed_diff, perm_diff, p_value




def compute_permutation_effect_size(data: pd.DataFrame, n_perm: int) -> dict : 
    
    beta_on = data[
        (data["condition"] == "Beta") &
        (data["laser_state"] == "LaserOn")
    ]
    beta_off = data[data["laser_state"] == "LaserOff"]

    conti_on = data[
        (data["condition"] == "Conti") &
        (data["laser_state"] == "LaserOn")
    ]
    conti_off = data[data["laser_state"] == "LaserOff"]

    # --- Permutation test ---
    b_observed_diff, b_perm_diff, b_pval = permutation(beta_on, beta_off, n_perm)
    c_observed_diff, c_perm_diff, c_pval = permutation(conti_on, conti_off, n_perm)
    bc_observed_diff, bc_perm_diff, bc_pval = permutation(conti_on, beta_on, n_perm)

    bc_cohen = pg.compute_effsize(b_perm_diff, c_perm_diff)
    b_cohen = pg.compute_effsize(b_perm_diff, bc_perm_diff)
    c_cohen = pg.compute_effsize(c_perm_diff, bc_perm_diff)

    results = [
        {
            "Condition" : "Beta vs NOstim",
            "observed mean difference" : b_observed_diff,
            "permutation differences" : b_perm_diff,
            "p-value" : b_pval,
            "cohen" : b_cohen
        },
        {
            "Condition" : "Conti vs NOstim",
            "observed mean difference" : c_observed_diff,
            "permutation differences" : c_perm_diff,
            "p-value" : c_pval,
            "cohen" : c_cohen
        },
        {
            "Condition" : "Beta vs Conti",
            "observed mean difference" : bc_observed_diff,
            "permutation differences" : bc_perm_diff,
            "p-value" : bc_pval,
            "cohen" : bc_cohen
        }
    ]

    res = pd.DataFrame(results)
    df = res.set_index("Condition")
    print(df[["observed mean difference", "p-value", "cohen"]].T)

    return results

    