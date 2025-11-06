# Przewidywanie naciekania pozatorebkowego (EPE) na podstawie badania MRI i biopsji

## Dane
Wykorzystano dane z pliku `baza zanonimizowana UZUPEŁNIONA.xlsx`.

Użyte kolumny:
- `wiek`
- `PSA`
- `PSAdensity`
- `MRI vol`
- `MRI SIZE`
- `MRI Pirads`
- `MRI EPE`
- `MRI EPE L`
- `MRI EPE P`
- `MRI SVI`
- `MRI SVI L`
- `MRI SVI P`
- `Bx ISUP Grade P`
- `Bx ISUP Grade L`
- `Bx ISUP Grade`

Przewidywana kolumna: `EPE RP`

Testowany model: `XGBoost` (zoptymalizowany pod kątem brier score)

Stosunek negatywnych do pozytywnych przypadków naciekania pozatorebkowego EPE:

![alt text](raport_epe_files/epe_dist.png)


## Wyniki
**Krzywa ROC:**

![alt text](raport_epe_files/epe_roc.png)

**Najbardziej wartościowe kolumny dla modelu:**

![alt text](raport_epe_files/epe_importances.png)

**Metryki**

<table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.22</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>0.7692</td>
        <td rowspan="4">
            <img src="raport_epe_files/epe_conf_022.png" alt="Confusion Matrix at 0.22 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.7500</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.7561</td>
    </tr>

</table>


<table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.1</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>1.0000</td>
        <td rowspan="4">
            <img src="raport_epe_files/epe_conf_01.png" alt="Confusion Matrix at 0.1 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.3214</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.5366</td>
    </tr>

</table>

<table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.4</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>0.4615</td>
        <td rowspan="4">
            <img src="raport_epe_files/epe_conf_04.png" alt="Confusion Matrix at 0.4 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.8929</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.7561</td>
    </tr>

</table>

# Przewidywanie makroskopowego naciekania pozatorebkowego (Macroscopic EPE) na podstawie badania MRI i biopsji

## Dane
Wykorzystano dane z pliku `baza zanonimizowana UZUPEŁNIONA.xlsx`.

Użyte kolumny:
- `wiek`
- `PSA`
- `PSAdensity`
- `MRI vol`
- `MRI SIZE`
- `MRI Pirads`
- `MRI EPE`
- `MRI EPE L`
- `MRI EPE P`
- `MRI SVI`
- `MRI SVI L`
- `MRI SVI P`
- `Bx ISUP Grade P`
- `Bx ISUP Grade L`
- `Bx ISUP Grade`

Przewidywana kolumna: `EPE macro` (utworzona kolumna, która posiada wartość 1 jeśli `EPE RP` == 1 oraz `MRI EPE` == 1, w przeciwnym wypadku 0)

Testowany model: `XGBoost` (zoptymalizowany pod kątem brier score)

Stosunek negatywnych do pozytywnych przypadków naciekania makroskopowego:

![alt text](raport_epe_files/macro_epe_dist.png)


## Wyniki
**Krzywa ROC:**

![alt text](raport_epe_files/macro_roc.png)

**Najbardziej wartościowe kolumny dla modelu:**

![alt text](raport_epe_files/macro_importances.png)

**Metryki**

<table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.03</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>1.0000</td>
        <td rowspan="4">
            <img src="raport_epe_files/macro_conf_003.png" alt="Confusion Matrix at 0.03 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.9459</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.9506</td>
    </tr>

</table>


<table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.5</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>0.5714</td>
        <td rowspan="4">
            <img src="raport_epe_files/macro_conf_05.png" alt="Confusion Matrix at 0.5 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.9865</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.9506</td>
    </tr>

</table>


# Przewidywanie mikroskopowego naciekania pozatorebkowego (Microscopic EPE) na podstawie badania MRI i biopsji

## Dane
Wykorzystano dane z pliku `baza zanonimizowana UZUPEŁNIONA.xlsx`.

Użyte kolumny:
- `wiek`
- `PSA`
- `PSAdensity`
- `MRI vol`
- `MRI SIZE`
- `MRI Pirads`
- `MRI EPE`
- `MRI EPE L`
- `MRI EPE P`
- `MRI SVI`
- `MRI SVI L`
- `MRI SVI P`
- `Bx ISUP Grade P`
- `Bx ISUP Grade L`
- `Bx ISUP Grade`

Przewidywana kolumna: `EPE micro` (utworzona kolumna, która posiada wartość 1 jeśli `EPE RP` == 1 oraz `MRI EPE` == 0, w przeciwnym wypadku 0)

<!-- reduced_df['EPE micro'] = (((reduced_df['EPE RP'] == 1) & (reduced_df['MRI EPE (naciek poza torebke)'] == 0))).astype(float)
reduced_df['EPE macro'] = (((reduced_df['EPE RP'] == 1) & (reduced_df['MRI EPE (naciek poza torebke)'] == 1))).astype(float) -->

Testowany model: `XGBoost` (zoptymalizowany pod kątem brier score)

Stosuenk negatywnych do pozytywnych przypadków naciekania mikroskopowego:

![alt text](raport_epe_files/micro_epe_dist.png)


## Wyniki
**Krzywa ROC:**

![alt text](raport_epe_files/micro_roc.png)

**Najbardziej wartościowe kolumny dla modelu:**

![alt text](raport_epve_files/micro_importances.png)

**Metryki**

<table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.15</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>0.8947</td>
        <td rowspan="4">
            <img src="raport_epe_files/micro_conf_015.png" alt="Confusion Matrix at 0.15 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.5806</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.6543</td>
    </tr>

</table>


<table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.35</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>0.6316</td>
        <td rowspan="4">
            <img src="raport_epe_files/micro_conf_035.png" alt="Confusion Matrix at 0.35 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.8871</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.8272</td>
    </tr>

</table>

# Przewidywanie naciekania na węzły chłonne na podstawie badania MRI i biopsji

## Dane
Wykorzystano dane z pliku `baza zanonimizowana UZUPEŁNIONA.xlsx`.

Użyte kolumny:
- `wiek`
- `PSA`
- `PSAdensity`
- `MRI vol`
- `MRI SIZE`
- `MRI Pirads`
- `MRI EPE`
- `MRI EPE L`
- `MRI EPE P`
- `MRI SVI`
- `MRI SVI L`
- `MRI SVI P`
- `Bx ISUP Grade P`
- `Bx ISUP Grade L`
- `Bx ISUP Grade`

Przewidywana kolumna: `N+`

Testowany model: `XGBoost` (zoptymalizowany pod kątem AUC)

Stosuenk negatywnych do pozytywnych przypadków naciekania na węzły chłonne:

![alt text](img_n+/dystrybucja_n_plus.png)


## Wyniki
**Krzywa ROC:**

![alt text](img_n+/xgboost_grid_auc.png)


**Macierz błędów:**

<table>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>AUC</td>
        <td>0.853</td>
        <td rowspan="4">
            <img src="img_n+/xgboost_grid_conf_at_20.png" alt="Confusion Matrix at 0.2 threshold">
        </td>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>0.3750</td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.9726</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.9136</td>
    </tr>

</table>

--- 

Otrzymane AUC jest wysokie.

Wysoka ilość FN - model nie wykrywa niektórych przydapków pozytywnych.

Zmieniając próg decyzji możemy sterować tą wielkością kosztem zwiększenia FP, zwiększymy sensitivity ale spadnie zarówno accuracy jak i specificity.


<table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.1</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>0.5000</td>
        <td rowspan="4">
            <img src="img_n+/xgboost_grid_conf_at_10.png" alt="Confusion Matrix at 0.1 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.8767</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.8395</td>
    </tr>

</table>


<table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.074</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>0.8750</td>
        <td rowspan="4">
            <img src="img_n+/xgboost_grid_conf_at_74.png" alt="Confusion Matrix at 0.074 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.7945</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.8025</td>
    </tr>

</table>

## Najwzażniejsze kolumny wskazane przez model
1. `Bx ISUP Grade`
2. `PSA`
3. `MRI EPE P`
4. `MRI SVI`

**Wpływ poszczególnych kolumn na ostateczny wynik**

![alt text](./img_n+/feature_importance.png)

# Przewidywanie SVI RP na podstawie badania MRI i biopsji

## Dane
Wykorzystano dane z pliku `baza zanonimizowana UZUPEŁNIONA.xlsx`.

Użyte kolumny:
- `wiek`
- `PSA`
- `PSAdensity`
- `MRI vol`
- `MRI SIZE`
- `MRI Pirads`
- `MRI EPE`
- `MRI EPE L`
- `MRI EPE P`
- `MRI SVI`
- `MRI SVI L`
- `MRI SVI P`
- `Bx ISUP Grade P`
- `Bx ISUP Grade L`
- `Bx ISUP Grade`

Przewidywana kolumna: `SVI`

Testowany model: `XGBoost` (zoptymalizowany pod kątem AUC)

Stosuenk negatywnych do pozytywnych przypadków SVI:

![alt text](./raport_svi/svi_dist.png)


## Wyniki
**Krzywa ROC:**

![alt text](./raport_svi/svi_auc.png)

<table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.27</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>0.3000</td>
        <td rowspan="4">
            <img src="./raport_svi/svi_conf_27.png" alt="Confusion Matrix at 0.27 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.9296</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.8519</td>
    </tr>

</table>


<table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.61</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>0.2000</td>
        <td rowspan="4">
            <img src="./raport_svi/svi_conf_61.png" alt="Confusion Matrix at 0.61 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.9577</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.8642</td>
    </tr>

</table>

<table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.1</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>0.4000</td>
        <td rowspan="4">
            <img src="./raport_svi/svi_conf_10.png" alt="Confusion Matrix at 0.1 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.8592</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.8025</td>
    </tr>

</table>

## Feature inportances

![alt text](./raport_svi/svi_fe.png)

# Przewidywanie naciekania pozatorebkowego na podstawie badania MRI (w późniejszym etapie dodatkowo gdzie celować biopsję).

## Dane
Wykorzystano dane z pliku `baza zanonimizowana UZUPEŁNIONA.xlsx`.

Użyte kolumny:
- `wiek`
- `PSA`
- `MRI vol`
- `MRI SIZE`
- `MRI Pirads`
- `MRI EPE (naciek poza torebke)`
- `MRI SVI (pęcherzyki)`
- `Bx ISUP Grade` (do wcześniejszej analizy, nie modelu)

Przewidywana kolumna: `EPE RP`

Testowany model: `XGBoost` (zoptymalizowany pod kątem AUC)

Stosunek negatywnych do pozytywnych przypadków naciekania pozatorebkowego:

![alt text](zdjecia_tosi/dystrybucja_epe_rp.png)


## Wyniki
**Krzywa ROC:**

![alt text](zdjecia_tosi/roc_curve.png)


**Macierz błędów:**

![alt text](zdjecia_tosi/confusion_matrix.png)

**Metryki**
Metryka | wynik
:---|:---
AUC | 0.80
Sensitivity | 0.6538
Specificity | 0.80
Accuracy | 0.7531

Otrzymane AUC jest w porządku.

Zmieniając próg decyzji możemy sterować tą wielkością kosztem zwiększenia FP, zwiększymy sensitivity ale spadnie zarówno accuracy jak i specificity.


<!-- <table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.1</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>0.5000</td>
        <td rowspan="4">
            <img src="xgboost_grid_conf_at_10.png" alt="Confusion Matrix at 0.1 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.8767</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.8395</td>
    </tr>

</table>


<table>
    <tr>
        <th colspan="3" style="text-align:center">Próg decyzji 0.074</th>
    </tr>
    <tr>
        <th style="text-align:center">Metryka</th>
        <th style="text-align:center">Wynik</th>
        <th style="text-align:center">Macierz błędów</th>
    </tr>
    </tr>
    <tr>
        <td>Sensitivity</td>
        <td>0.8750</td>
        <td rowspan="4">
            <img src="xgboost_grid_conf_at_74.png" alt="Confusion Matrix at 0.074 threshold">
        </td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td>0.7945</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>0.8025</td>
    </tr>

</table> -->