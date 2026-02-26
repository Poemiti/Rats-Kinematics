#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd




def ANOVA(data: pd.DataFrame, formula: str = "velocity ~ condition * laser_state * laser_intensity"):

    from rats_kinematics_utils.plot_comparative import _trim_extremes_iqr

    data_trimmed = _trim_extremes_iqr(data, k=1.5).copy()

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



def check_normality(data) -> bool : 
    from scipy.stats import shapiro

    stat, pval = shapiro(data)
    print(f"Statistic: {stat}, p-value: {pval}")

    if pval < 0.05 : 
        print("Normality = True")
        return True
    
    print("Normality = False")
    return False


