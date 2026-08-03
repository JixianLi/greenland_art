# The 155 MAR columns

Every column of `datasets/mar/mar_training_2000_2009.npz` (`features`, shape
3,374,732 x 155), with the meaning and units taken from the source NetCDF
attributes rather than restated from memory.

**Provenance.** MAR v3.2 regional climate model, NCEPv1-forced, 20 km grid,
Greenland, 2000-2009, every 5th day. This is *model output, not observation*.
One row is one ice-sheet grid cell on one day: 4,604 cells with `MSK >= 50 %`,
74 days a year, 10 years, minus rows carrying non-finite values.

## Reading the `_Lnn` suffix

The suffix is a level index into whichever vertical (or sector) axis the field
carries, and **the axis differs by field family**. It is always the index in the
*original* file, so it stays valid even where later columns were dropped.

| axis | size | values | used by |
|---|---|---|---|
| `SECTOR` | 2 | 1 = ice sheet, 2 = tundra | sliced to sector 1, **no suffix added** |
| `SECTOR1_1` | 1 | 1 | suffix `_L00`, carries no information |
| `OUTLAY` | 18 | 0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20 m **below the surface** | `RO1`, `TI1`, `WA1` |
| `ATMLAY` | 3 | sigma 0.99849, 0.99924, 0.99962 (near-surface) | `TT`, `QQ`, `RH`, `UU`, `VV`, `UV`, `ZZ` |
| `ATMLAY3_3` | 1 | sigma 0.99962 | `TTMIN`, `TTMAX`, `UVMAX` |
| `ZTQLEV` | 5 | 2, 10, 25, 50, 100 m above ground | `TTZ`, `QQZ` |
| `ZUVLEV` | 4 | 10, 25, 50, 100 m above ground | `UUZ`, `VVZ`, `UVZ` |
| `PLEV` | 7 | 200, 500, 600, 700, 800, 850, 925 hPa | `TTP`, `QQP`, `UUP`, `VVP`, `ZZP` |

Two traps in that table. A size-1 `SECTOR1_1` axis is not excluded the way
`SECTOR` is, so `ME`, `RZ`, `SW`, `SWSN`, `G11`, `G21`, `SHSN0` and `SHSN2`
appear as `_L00` while `SMB`, `SU`, `RU` and friends appear bare -- the suffix
does not mean what it looks like it means. And `OUTLAY` is a **depth in metres**,
not a layer counter: the firn profile spans 0 to 20 m below the surface.

## Columns that are not in the matrix

`PLEV` indices 3-6 (700, 800, 850, 925 hPa) are dropped for all five pressure-
level fields -- 20 columns. Those pressure surfaces lie below the ice surface
over most of the interior, where MAR writes a fill value. 600 hPa is roughly
4,200 m, clear of the 3,200 m summit; 700 hPa is roughly 3,000 m and is not.

## Redundancy worth knowing before reading an explained-variance number

Several columns are near-copies of each other, measured over ice cells in 2009:

| pair | R^2 |
|---|---|
| `ST` / `ST2` (two surface temperatures) | 0.9989 |
| `TT_L02` (sigma 0.99962) / `TTZ_L00` (2 m) | 0.9989 |
| `AL1` / `AL2` (two of the three albedos) | 0.9827 |
| `UV_L02` / `UVZ_L00` (10 m wind) | 0.9165 |
| `AL` / `AL2` | 0.8622 |

A duplicated column is free variance for any compressor, so it flatters both PCA
and the autoencoder equally. It does not invalidate the comparison between them,
which is the point of that figure, but it does mean "97.6 % of the variance" is
not the same claim as "97.6 % of the physics".

`ZZ` deserves a line of its own. Its `long_name` is *Model Surface Height*, and
`ZZ_L00` matches the surface elevation field at R^2 = 1.0000 (`ZZ_L00` vs
`ZZ_L02`, also 1.0000; `SP` vs elevation, 0.9942). Surface elevation is an input
to this model in all but name -- see `scripts/plot_latent_comparison.py`.

## All 155 columns

| # | column | meaning | units | level |
|---|---|---|---|---|
| 1 | `SF` | Snowfall | mmWE/day | surface |
| 2 | `RF` | Rainfall | mmWE/day | surface |
| 3 | `CP` | Convective precipitation | mmWE/day | surface |
| 4 | `ME_L00` | Meltwater production | mmWE/day | single (SECTOR1_1) |
| 5 | `RZ_L00` | Meltwater refreezing and deposition | mmWE/day | single (SECTOR1_1) |
| 6 | `SW_L00` | Surface Water | mmWE/day | single (SECTOR1_1) |
| 7 | `SWSN_L00` | Surficial Water Specific Mass | mmWE | single (SECTOR1_1) |
| 8 | `SWD` | Short Wave Downward | W/m2 | surface |
| 9 | `LWD` | Long Wave Downward | W/m2 | surface |
| 10 | `LWU` | Long Wave Upward | W/m2 | surface |
| 11 | `SHF` | Sensible Heat Flux | W/m2 | surface |
| 12 | `LHF` | Latent Heat Flux | W/m2 | surface |
| 13 | `AL` | Albedo | - | surface |
| 14 | `AL2` | Albedo | - | sector 1 (ice sheet) |
| 15 | `ST` | Surface Temperature | C | surface |
| 16 | `PDD` | Postive Degree Day | C | surface |
| 17 | `SP` | Surface Pressure | hPa | surface |
| 18 | `TTMIN_L00` | Min Temp | C | single (ATMLAY3_3) |
| 19 | `TTMAX_L00` | Max Temp | C | single (ATMLAY3_3) |
| 20 | `UVMAX_L00` | Maximum Wind Speed | m/s | single (ATMLAY3_3) |
| 21 | `SHSN0_L00` | Ini. old firn/ice thickness | m | single (SECTOR1_1) |
| 22 | `SHSN2_L00` | Snow Pack Height above Ice | m | single (SECTOR1_1) |
| 23 | `G11_L00` | g1 (Dendri./Spheri.) | - | single (OUTLAY1_1) |
| 24 | `G21_L00` | g2 (Sphericity/Size) | - | single (OUTLAY1_1) |
| 25 | `QW` | Cloud Dropplets Concent. | kg/kg | surface |
| 26 | `QI` | Cloud Ice Crystals Concent. | kg/kg | surface |
| 27 | `QS` | Cloud Snow Flakes Concent. | kg/kg | surface |
| 28 | `QR` | Cloud Rain Concentration | kg/kg | surface |
| 29 | `CC` | Cloud Cover | - | surface |
| 30 | `COD` | Cloud Optical Depth | - | surface |
| 31 | `CU` | Cloud Cover (up) | - | surface |
| 32 | `CM` | Cloud Cover (Middle) | - | surface |
| 33 | `CD` | Cloud Cover (down) | - | surface |
| 34 | `WVP` | Water Vapour Path | kg/m2 | surface |
| 35 | `IWP` | Ice Water Path | kg/m2 | surface |
| 36 | `CWP` | Condensed Water Path | kg/m2 | surface |
| 37 | `SMB` | Surface Mass Balance (SMB~SF+RF-RU-SU-SW) | mmWE/day | sector 1 (ice sheet) |
| 38 | `SU` | Sublimation and evaporation | mmWE/day | sector 1 (ice sheet) |
| 39 | `RU` | Run-off of meltwater and rain water | mmWE/day | sector 1 (ice sheet) |
| 40 | `AL1` | Albedo (Tot Refl/Tot Inc) | - | sector 1 (ice sheet) |
| 41 | `SHSN3` | Snow Pack Height Total | m | sector 1 (ice sheet) |
| 42 | `ST2` | Surface Temperature | C | sector 1 (ice sheet) |
| 43 | `PBL` | Height of Bound. Layer (2val.) | m | sector 1 (ice sheet) |
| 44 | `RO1_L00` | Snow Density | kg/m3 | 0 m depth |
| 45 | `RO1_L01` | Snow Density | kg/m3 | 0.05 m depth |
| 46 | `RO1_L02` | Snow Density | kg/m3 | 0.1 m depth |
| 47 | `RO1_L03` | Snow Density | kg/m3 | 0.2 m depth |
| 48 | `RO1_L04` | Snow Density | kg/m3 | 0.3 m depth |
| 49 | `RO1_L05` | Snow Density | kg/m3 | 0.4 m depth |
| 50 | `RO1_L06` | Snow Density | kg/m3 | 0.5 m depth |
| 51 | `RO1_L07` | Snow Density | kg/m3 | 0.65 m depth |
| 52 | `RO1_L08` | Snow Density | kg/m3 | 0.8 m depth |
| 53 | `RO1_L09` | Snow Density | kg/m3 | 1 m depth |
| 54 | `RO1_L10` | Snow Density | kg/m3 | 1.5 m depth |
| 55 | `RO1_L11` | Snow Density | kg/m3 | 2 m depth |
| 56 | `RO1_L12` | Snow Density | kg/m3 | 3 m depth |
| 57 | `RO1_L13` | Snow Density | kg/m3 | 5 m depth |
| 58 | `RO1_L14` | Snow Density | kg/m3 | 7.5 m depth |
| 59 | `RO1_L15` | Snow Density | kg/m3 | 10 m depth |
| 60 | `RO1_L16` | Snow Density | kg/m3 | 15 m depth |
| 61 | `RO1_L17` | Snow Density | kg/m3 | 20 m depth |
| 62 | `TI1_L00` | Ice/Snow Temperature | C | 0 m depth |
| 63 | `TI1_L01` | Ice/Snow Temperature | C | 0.05 m depth |
| 64 | `TI1_L02` | Ice/Snow Temperature | C | 0.1 m depth |
| 65 | `TI1_L03` | Ice/Snow Temperature | C | 0.2 m depth |
| 66 | `TI1_L04` | Ice/Snow Temperature | C | 0.3 m depth |
| 67 | `TI1_L05` | Ice/Snow Temperature | C | 0.4 m depth |
| 68 | `TI1_L06` | Ice/Snow Temperature | C | 0.5 m depth |
| 69 | `TI1_L07` | Ice/Snow Temperature | C | 0.65 m depth |
| 70 | `TI1_L08` | Ice/Snow Temperature | C | 0.8 m depth |
| 71 | `TI1_L09` | Ice/Snow Temperature | C | 1 m depth |
| 72 | `TI1_L10` | Ice/Snow Temperature | C | 1.5 m depth |
| 73 | `TI1_L11` | Ice/Snow Temperature | C | 2 m depth |
| 74 | `TI1_L12` | Ice/Snow Temperature | C | 3 m depth |
| 75 | `TI1_L13` | Ice/Snow Temperature | C | 5 m depth |
| 76 | `TI1_L14` | Ice/Snow Temperature | C | 7.5 m depth |
| 77 | `TI1_L15` | Ice/Snow Temperature | C | 10 m depth |
| 78 | `TI1_L16` | Ice/Snow Temperature | C | 15 m depth |
| 79 | `TI1_L17` | Ice/Snow Temperature | C | 20 m depth |
| 80 | `WA1_L00` | Liquid Water Content | kg/kg | 0 m depth |
| 81 | `WA1_L01` | Liquid Water Content | kg/kg | 0.05 m depth |
| 82 | `WA1_L02` | Liquid Water Content | kg/kg | 0.1 m depth |
| 83 | `WA1_L03` | Liquid Water Content | kg/kg | 0.2 m depth |
| 84 | `WA1_L04` | Liquid Water Content | kg/kg | 0.3 m depth |
| 85 | `WA1_L05` | Liquid Water Content | kg/kg | 0.4 m depth |
| 86 | `WA1_L06` | Liquid Water Content | kg/kg | 0.5 m depth |
| 87 | `WA1_L07` | Liquid Water Content | kg/kg | 0.65 m depth |
| 88 | `WA1_L08` | Liquid Water Content | kg/kg | 0.8 m depth |
| 89 | `WA1_L09` | Liquid Water Content | kg/kg | 1 m depth |
| 90 | `WA1_L10` | Liquid Water Content | kg/kg | 1.5 m depth |
| 91 | `WA1_L11` | Liquid Water Content | kg/kg | 2 m depth |
| 92 | `WA1_L12` | Liquid Water Content | kg/kg | 3 m depth |
| 93 | `WA1_L13` | Liquid Water Content | kg/kg | 5 m depth |
| 94 | `WA1_L14` | Liquid Water Content | kg/kg | 7.5 m depth |
| 95 | `WA1_L15` | Liquid Water Content | kg/kg | 10 m depth |
| 96 | `WA1_L16` | Liquid Water Content | kg/kg | 15 m depth |
| 97 | `WA1_L17` | Liquid Water Content | kg/kg | 20 m depth |
| 98 | `TT_L00` | Temperature | C | 0.998488 sigma |
| 99 | `TT_L01` | Temperature | C | 0.999244 sigma |
| 100 | `TT_L02` | Temperature | C | 0.999622 sigma |
| 101 | `QQ_L00` | Specific Humidity | g/kg | 0.998488 sigma |
| 102 | `QQ_L01` | Specific Humidity | g/kg | 0.999244 sigma |
| 103 | `QQ_L02` | Specific Humidity | g/kg | 0.999622 sigma |
| 104 | `RH_L00` | Relative Humidity | % | 0.998488 sigma |
| 105 | `RH_L01` | Relative Humidity | % | 0.999244 sigma |
| 106 | `RH_L02` | Relative Humidity | % | 0.999622 sigma |
| 107 | `UU_L00` | x-Wind Speed component | m/s | 0.998488 sigma |
| 108 | `UU_L01` | x-Wind Speed component | m/s | 0.999244 sigma |
| 109 | `UU_L02` | x-Wind Speed component | m/s | 0.999622 sigma |
| 110 | `VV_L00` | y-Wind Speed component | m/s | 0.998488 sigma |
| 111 | `VV_L01` | y-Wind Speed component | m/s | 0.999244 sigma |
| 112 | `VV_L02` | y-Wind Speed component | m/s | 0.999622 sigma |
| 113 | `UV_L00` | Wind Speed | m/s | 0.998488 sigma |
| 114 | `UV_L01` | Wind Speed | m/s | 0.999244 sigma |
| 115 | `UV_L02` | Wind Speed | m/s | 0.999622 sigma |
| 116 | `ZZ_L00` | Model Surface Height | m | 0.998488 sigma |
| 117 | `ZZ_L01` | Model Surface Height | m | 0.999244 sigma |
| 118 | `ZZ_L02` | Model Surface Height | m | 0.999622 sigma |
| 119 | `TTZ_L00` | Temperature | C | 2 m AGL |
| 120 | `TTZ_L01` | Temperature | C | 10 m AGL |
| 121 | `TTZ_L02` | Temperature | C | 25 m AGL |
| 122 | `TTZ_L03` | Temperature | C | 50 m AGL |
| 123 | `TTZ_L04` | Temperature | C | 100 m AGL |
| 124 | `QQZ_L00` | Specific Humidity | g/kg | 2 m AGL |
| 125 | `QQZ_L01` | Specific Humidity | g/kg | 10 m AGL |
| 126 | `QQZ_L02` | Specific Humidity | g/kg | 25 m AGL |
| 127 | `QQZ_L03` | Specific Humidity | g/kg | 50 m AGL |
| 128 | `QQZ_L04` | Specific Humidity | g/kg | 100 m AGL |
| 129 | `UUZ_L00` | x-Wind Speed component | m/s | 10 m AGL |
| 130 | `UUZ_L01` | x-Wind Speed component | m/s | 25 m AGL |
| 131 | `UUZ_L02` | x-Wind Speed component | m/s | 50 m AGL |
| 132 | `UUZ_L03` | x-Wind Speed component | m/s | 100 m AGL |
| 133 | `VVZ_L00` | y-Wind Speed component | m/s | 10 m AGL |
| 134 | `VVZ_L01` | y-Wind Speed component | m/s | 25 m AGL |
| 135 | `VVZ_L02` | y-Wind Speed component | m/s | 50 m AGL |
| 136 | `VVZ_L03` | y-Wind Speed component | m/s | 100 m AGL |
| 137 | `UVZ_L00` | Horizontal Wind Speed | m/s | 10 m AGL |
| 138 | `UVZ_L01` | Horizontal Wind Speed | m/s | 25 m AGL |
| 139 | `UVZ_L02` | Horizontal Wind Speed | m/s | 50 m AGL |
| 140 | `UVZ_L03` | Horizontal Wind Speed | m/s | 100 m AGL |
| 141 | `TTP_L00` | Temperature | C | 200 hPa |
| 142 | `TTP_L01` | Temperature | C | 500 hPa |
| 143 | `TTP_L02` | Temperature | C | 600 hPa |
| 144 | `QQP_L00` | Specific Humidity | g/kg | 200 hPa |
| 145 | `QQP_L01` | Specific Humidity | g/kg | 500 hPa |
| 146 | `QQP_L02` | Specific Humidity | g/kg | 600 hPa |
| 147 | `UUP_L00` | x-Wind Speed component | m/s | 200 hPa |
| 148 | `UUP_L01` | x-Wind Speed component | m/s | 500 hPa |
| 149 | `UUP_L02` | x-Wind Speed component | m/s | 600 hPa |
| 150 | `VVP_L00` | y-Wind Speed component | m/s | 200 hPa |
| 151 | `VVP_L01` | y-Wind Speed component | m/s | 500 hPa |
| 152 | `VVP_L02` | y-Wind Speed component | m/s | 600 hPa |
| 153 | `ZZP_L00` | Height | m | 200 hPa |
| 154 | `ZZP_L01` | Height | m | 500 hPa |
| 155 | `ZZP_L02` | Height | m | 600 hPa |
