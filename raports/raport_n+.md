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