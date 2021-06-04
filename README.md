- [Data Scheme](#data-scheme)
- [Folder Structure](#folder-structure)
- [Notes](#notes)
  

## Data Scheme

Formato Righe:

`5.9,0.645,0.12,2,0.075,32,44,0.99547,3.57,0.71,10.2,0` ( `\n` )

Attributi:

|   N   | Valore                   |
| :---: | :----------------------- |
|   1   | Fixed Acidity            |
|   2   | Volatile Acidity         |
|   3   | Citric Acidity           |
|   4   | Residual Sugar           |
|   5   | Chlorides                |
|   6   | Free Sulfur Dioxide      |
|   7   | Total Sulfur Dioxide     |
|   8   | Density                  |
|   9   | pH                       |
|  10   | Sulphates                |
|  11   | Alcohol                  |
|  12   | **Quality** (0 < L < 10) |


## Folder Structure

```
|--root
    |--data (Train, Test, Models..)
    |--code (Actual Code)
    |--img  (Plots of Results)
    |--junk (Temporary Files)
    |--report   (TeX files, report goes in root)
```

## Notes
* Some features ar gaussian-ish, some not (See `dist.jpg`)
* Classes are note equally balaced