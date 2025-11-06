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

![alt text](raport_epe_files/micro_importances.png)

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
