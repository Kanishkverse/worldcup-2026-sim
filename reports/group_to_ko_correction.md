# Group-to-knockout correction

Player-aggregated ratings, 15 played group games, per-game physics at 250 sims each. Residuals become Elo corrections carried into the elimination rounds.

## Predicted vs actual, game by game

| Match | Predicted GD | Actual | Residual (home) |
|---|---|---|---|
| Mexico 2-0 South Africa | +0.72 | +2 | +1.28 |
| South Korea 2-1 Czechia | -0.04 | +1 | +1.04 |
| Canada 1-1 Bosnia and Herzegovina | +0.31 | +0 | -0.31 |
| Qatar 1-1 Switzerland | -1.14 | +0 | +1.14 |
| Brazil 1-1 Morocco | +0.04 | +0 | -0.04 |
| Haiti 0-1 Scotland | -0.96 | -1 | -0.04 |
| United States 4-1 Paraguay | +0.26 | +3 | +2.74 |
| Australia 2-0 Türkiye | -0.47 | +2 | +2.47 |
| Germany 7-1 Curaçao | +1.31 | +6 | +4.69 |
| Côte d'Ivoire 1-0 Ecuador | -0.29 | +1 | +1.29 |
| Netherlands 2-2 Japan | +0.22 | +0 | -0.22 |
| Sweden 5-1 Tunisia | +0.32 | +4 | +3.68 |
| Belgium 1-1 Egypt | +0.50 | +0 | -0.50 |
| Spain 0-0 Cabo Verde | +1.66 | +0 | -1.66 |
| Saudi Arabia 1-1 Uruguay | -0.77 | +0 | +0.77 |

## Learned corrections (teams with games played)

| Team | Games | Mean residual | Correction (Elo) | Player Elo | Corrected |
|---|---|---|---|---|---|
| United States | 1 | +2.74 | +110 | 1863 | 1973 |
| Australia | 1 | +2.47 | +110 | 1712 | 1822 |
| Germany | 1 | +4.69 | +110 | 1960 | 2070 |
| Sweden | 1 | +3.68 | +110 | 1805 | 1915 |
| Cabo Verde | 1 | +1.66 | +75 | 1608 | 1682 |
| Côte d'Ivoire | 1 | +1.29 | +58 | 1753 | 1811 |
| Mexico | 1 | +1.28 | +57 | 1797 | 1854 |
| Qatar | 1 | +1.14 | +51 | 1532 | 1583 |
| South Korea | 1 | +1.04 | +47 | 1781 | 1828 |
| Saudi Arabia | 1 | +0.77 | +35 | 1634 | 1669 |
| Egypt | 1 | +0.50 | +23 | 1700 | 1723 |
| Bosnia and Herzegovina | 1 | +0.31 | +14 | 1680 | 1694 |
| Japan | 1 | +0.22 | +10 | 1903 | 1913 |
| Scotland | 1 | +0.04 | +2 | 1804 | 1806 |
| Morocco | 1 | +0.04 | +2 | 1920 | 1922 |
| Brazil | 1 | -0.04 | -2 | 1965 | 1963 |
| Haiti | 1 | -0.04 | -2 | 1532 | 1531 |
| Netherlands | 1 | -0.22 | -10 | 2005 | 1996 |
| Canada | 1 | -0.31 | -14 | 1795 | 1781 |
| Belgium | 1 | -0.50 | -23 | 1899 | 1876 |
| Uruguay | 1 | -0.77 | -35 | 1916 | 1881 |
| Czechia | 1 | -1.04 | -47 | 1768 | 1721 |
| Switzerland | 1 | -1.14 | -51 | 1910 | 1859 |
| South Africa | 1 | -1.28 | -57 | 1584 | 1527 |
| Ecuador | 1 | -1.29 | -58 | 1899 | 1841 |
| Spain | 1 | -1.66 | -75 | 2135 | 2060 |
| Paraguay | 1 | -2.74 | -110 | 1766 | 1656 |
| Türkiye | 1 | -2.47 | -110 | 1902 | 1792 |
| Curaçao | 1 | -4.69 | -110 | 1521 | 1411 |
| Tunisia | 1 | -3.68 | -110 | 1671 | 1561 |

## Effect on the elimination rounds

Title and deep-round odds before and after the correction, for the teams that moved most.

| Team | Champ before | Champ after | SF before | SF after |
|---|---|---|---|---|
| Spain | 15.6% | 6.7% | 37.9% | 24.8% |
| Germany | 3.6% | 10.1% | 15.8% | 28.6% |
| United States | 1.7% | 7.4% | 12.0% | 26.5% |
| France | 15.4% | 13.0% | 38.7% | 34.0% |
| Sweden | 0.7% | 3.0% | 5.4% | 13.0% |
| Argentina | 11.3% | 13.3% | 34.6% | 35.9% |
| Switzerland | 2.3% | 0.9% | 11.8% | 6.4% |
| Uruguay | 2.1% | 0.8% | 12.9% | 8.6% |
| Türkiye | 1.1% | 0.1% | 4.8% | 1.2% |
| Netherlands | 4.8% | 3.8% | 19.3% | 17.0% |
| Colombia | 2.4% | 3.2% | 13.2% | 16.0% |
| Ecuador | 1.4% | 0.6% | 8.5% | 5.2% |
