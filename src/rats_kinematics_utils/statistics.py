#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

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


def cohens_d_rat_level(effects):
    return np.mean(effects) / np.std(effects, ddof=1)


def compute_rat_effects(df: pd.DataFrame,
                        value_col="value",
                        rat_col="rat",
                        laser_col="laser_state",
                        on_label="LaserOn",
                        off_label="LaserOff") -> pd.DataFrame :
    """
    Compute mean(ON) - mean(OFF) per rat.
    Returns dataframe of rat-level effects.
    """
    
    effects = []
    
    for rat, subdf in df.groupby([rat_col]):
        
        on_values = subdf[subdf[laser_col] == on_label][value_col]
        off_values = subdf[subdf[laser_col] == off_label][value_col]
        
        # Skip rats missing one condition
        if len(on_values) == 0 or len(off_values) == 0:
            continue
        
        effect = on_values.mean() - off_values.mean()
        effects.append({
            "rat" : rat,
            "effect" : effect
        })
    
    return pd.DataFrame(effects)





def rat_level_permutation(effects, n_perm=10000, two_tailed=True):
    """
    Sign-flip permutation test at rat level.
    """
    
    observed_mean = np.mean(effects)
    perm_means = []
    
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=len(effects))
        perm_means.append(np.mean(effects * signs))
    
    perm_means = np.array(perm_means)
    
    if two_tailed:
        p_value = np.mean(np.abs(perm_means) >= np.abs(observed_mean))
    else:
        p_value = np.mean(perm_means >= observed_mean)
    
    return observed_mean, p_value, perm_means





def compute_permutation_effect_size(data: pd.DataFrame) -> dict : 

    effects = compute_rat_effects(data)
    
    print("\nRat-level effects:\n", effects)
    print("Mean effect:", np.mean(effects["effect"]))
    
    observed, p_value, perm_dist = rat_level_permutation(effects["effect"])
    
    print("\nPermutation results")
    print("Observed mean effect:", observed)
    print("Permutation p-value:", p_value)
    
    if len(effects) > 1:
        d = cohens_d_rat_level(effects["effect"])
        print("Cohen's d (rat level):", d)
    else:
        print("Not enough rats for effect size.")
    
    return {
        "effects": effects,
        "observed_mean": observed,
        "p_value": p_value,
        "perm_distribution": perm_dist
    }
