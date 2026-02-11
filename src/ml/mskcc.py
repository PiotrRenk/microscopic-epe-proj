import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series


COEFFICIENTS = {
    "EPE": {
        "intercept": -4.3619472,
        "age": 0.02789286,
        "psa": 0.25391759,
        "psa_spline_1": -0.00185119,
        "psa_spline_2": 0.00512636,
        "biopsy_gleason": {2: 0.9456123, 3: 1.38600414, 4: 1.47229518, 5: 2.52672521},
        "clinical_stage": {
            "2a": 0.29223705,
            "2b": 0.8045909,
            "2c": 0.85209752,
            "3plus": 1.67265063,
        },
    },
    "EPE cores": {
        "intercept": -4.14615912,
        "age": 0.03263727,
        "psa": 0.22419499,
        "psa_spline_1": -0.00151357,
        "psa_spline_2": 0.0041806,
        "biopsy_gleason": {2: 0.62975509, 3: 1.04483516, 4: 1.11988728, 5: 2.04021401},
        "clinical_stage": {
            "2a": 0.21016394,
            "2b": 0.7631916,
            "2c": 0.56884638,
            "3plus": 1.46007476,
        },
        "no_positive_cores": 0.08760181,
        "no_negative_cores": -0.06353104,
    },
    "N+": {
        "intercept": -5.83057151,
        "age": 0.00521158,
        "psa": 0.18754729,
        "psa_spline_1": -0.00122617,
        "psa_spline_2": 0.0033653,
        "biopsy_gleason": {2: 1.52752948, 3: 2.57873595, 4: 2.75375893, 5: 3.50034615},
        "clinical_stage": {
            "2a": 0.26172062,
            "2b": 0.55860494,
            "2c": 0.84874365,
            "3plus": 1.09527926,
        },
    },
    "N+ cores": {
        "intercept": -5.37408605,
        "age": 0.01061319,
        "psa": 0.22266602,
        "psa_spline_1": -0.001599,
        "psa_spline_2": 0.00441175,
        "biopsy_gleason": {2: 0.998897, 3: 2.03362879, 4: 2.177025, 5: 2.87515732},
        "clinical_stage": {
            "2a": 0.17084652,
            "2b": 0.49919005,
            "2c": 0.46102494,
            "3plus": 0.74403104,
        },
        "no_positive_cores": 0.05972006,
        "no_negative_cores": -0.09466818,
    },
    "SVI": {
        "intercept": -5.96514874,
        "age": 0.01303056,
        "psa": 0.16554996,
        "psa_spline_1": -0.00095842,
        "psa_spline_2": 0.00262538,
        "biopsy_gleason": {2: 1.24764853, 3: 2.04455038, 4: 2.19455366, 5: 3.17089909},
        "clinical_stage": {
            "2a": 0.28661144,
            "2b": 0.58408599,
            "2c": 1.04372486,
            "3plus": 1.23384554,
        },
    },
    "SVI cores": {
        "intercept": -6.11427809,
        "age": 0.02483733,
        "psa": 0.1475095,
        "psa_spline_1": -0.00085465,
        "psa_spline_2": 0.00234133,
        "biopsy_gleason": {2: 1.11283762, 3: 1.89864418, 4: 2.06573755, 5: 2.97808178},
        "clinical_stage": {
            "2a": 0.13563953,
            "2b": 0.43790414,
            "2c": 0.67764059,
            "3plus": 0.80895957,
        },
        "no_positive_cores": 0.07682603,
        "no_negative_cores": -0.10972019,
    },
}


def calculate_splines(var):
    PSAPreopKnot1 = 0.2
    PSAPreopKnot2 = 4.8
    PSAPreopKnot3 = 7.35
    PSAPreopKnot4 = 307.0

    psa_spline_1 = (max(var - PSAPreopKnot1, 0) ** 3 - max(var - PSAPreopKnot3, 0) ** 3 * (PSAPreopKnot4 - PSAPreopKnot1) / (PSAPreopKnot4 - PSAPreopKnot3) + max(var - PSAPreopKnot4, 0) ** 3 * (PSAPreopKnot3 - PSAPreopKnot1) / (PSAPreopKnot4 - PSAPreopKnot3))
    psa_spline_2 = (max(var - PSAPreopKnot2, 0) ** 3 - max(var - PSAPreopKnot3, 0) ** 3 * (PSAPreopKnot4 - PSAPreopKnot2) / (PSAPreopKnot4 - PSAPreopKnot3) + max(var - PSAPreopKnot4, 0) ** 3 * (PSAPreopKnot3 - PSAPreopKnot2) / (PSAPreopKnot4 - PSAPreopKnot3))
    return psa_spline_1, psa_spline_2


def stage(tnm, stage_weights):
    tnm = tnm.lower()
    if "t2a" in tnm:
        return stage_weights["2a"]
    if "t2b" in tnm:
        return stage_weights["2b"]
    if "t2c" in tnm:
        return stage_weights["2c"]
    if "t3" in tnm or "t4" in tnm:
        return stage_weights["3plus"]
    return 0.0


def gleason(isup, gleason_weights):
    if isup == 2:
        return gleason_weights[2]
    if isup == 3:
        return gleason_weights[3]
    if isup == 4:
        return gleason_weights[4]
    if isup == 5:
        return gleason_weights[5]
    return 0.0


# TODO: update column names to the unified ones
def getCores(
    patient, split="na", column_P="ilość\xa0+ wycinków P", column_L="Ilość + wycinków L"
):
    cores_P = patient[column_P]
    if cores_P == 0:
        pos_P = 0
        neg_P = 0
    else:
        cores_P = cores_P.split(split)
        pos_P = int(cores_P[0])
        neg_P = int(cores_P[1]) - pos_P

    cores_L = patient[column_L]
    if cores_L == 0:
        pos_L = 0
        neg_L = 0
    else:
        cores_L = cores_L.split(split)
        pos_L = int(cores_L[0])
        neg_L = int(cores_L[1]) - pos_L

    return pos_P + pos_L, neg_P + neg_L

# TODO: update column names to the unified ones
def mskcc_predict(patient: Series, target: str = "EPE") -> float:
    psa_spline_1, psa_spline_2 = calculate_splines(patient["PSA"])

    b = (
        COEFFICIENTS[target]["intercept"]
        + COEFFICIENTS[target]["age"] * patient["wiek"]
        + COEFFICIENTS[target]["psa"] * patient["PSA"]
        + COEFFICIENTS[target]["psa_spline_1"] * psa_spline_1
        + COEFFICIENTS[target]["psa_spline_2"] * psa_spline_2
        + gleason(patient["Bx ISUP Grade"], COEFFICIENTS[target]["biopsy_gleason"])
        + stage(patient["TNM"], COEFFICIENTS[target]["clinical_stage"])
    )

    if "cores" in target:
        pos_cores, neg_cores = getCores(patient)
        b += (
            COEFFICIENTS[target]["no_positive_cores"] * pos_cores
            + COEFFICIENTS[target]["no_negative_cores"] * neg_cores
        )

    return np.exp(b) / (1 + np.exp(b))

def mskcc_predict_all(X: DataFrame, target: str = "EPE") -> NDArray:
    # return patient.apply(lambda x: mskcc_predict(x, target), axis=1)
    mskcc_probs = np.array([])
    for idx, patient in X.iterrows():        
        prob = mskcc_predict(patient, target=target)
        mskcc_probs = np.append(mskcc_probs, prob)

    return mskcc_probs