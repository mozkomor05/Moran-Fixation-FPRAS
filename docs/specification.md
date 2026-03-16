# Specifikace projektu

## Téma

Aproximace pravděpodobnosti fixace Moranova procesu na grafech.

## Popis

Moranův proces modeluje evoluci na populaci $N$ jedinců umístěných na vrcholech grafu. Jeden mutant s fitness $r$ soutěží proti $N-1$ rezidentům s fitness 1. V každém kroku (Birth-Death) se vybere reproduktor proporcionálně k fitness a nahradí náhodného souseda.

**Pravděpodobnost fixace** $\rho(G, r)$ je pravděpodobnost, že mutant ovládne celou populaci.

## Cíl

Implementace C++23 knihovny s Python rozhraním pro výpočet $\rho(G, r)$. Hlavním cílem je implementace tří úrovní FPRAS algoritmů s rostoucí efektivitou:

1. **Naivní MC** (Díaz et al. 2014, Thm 13) -- simuluje všechny BD kroky včetně neefektivních (oba uzly stejného typu). Absorpce na run: $O\bigl(\tfrac{r}{|r-1|}\, n^4\bigr)$, celková složitost: $O\bigl(\tfrac{r}{|r-1|}\, \tfrac{n^6}{\varepsilon^2}\bigr)$.
2. **Aktivní proces** (Chatterjee et al. 2017, Thm 11) -- počítá pouze efektivní kroky, kdy se počet mutantů skutečně změní. Celková složitost: $O(n^2 \Delta^2 \varepsilon^{-2} (\log n + \log \varepsilon^{-1}))$.
3. **Early termination** (Goldberg et al. 2019, Thm 5) -- rozšiřuje aktivní proces o včasné ukončení, když je fixace či vymření téměř jisté. Celková složitost: $O(\Delta^2 \bar{d}\, \varepsilon^{-2} \log(\bar{d}\, \varepsilon^{-1}))$, kde $\bar{d}$ je průměrný stupeň.

Dále:
- Exaktní vzorce pro úplné a regulární grafy (izotermální věta, Lieberman et al. 2005)
- Python bindings (pybind11, interoperabilita s NetworkX a SciPy)
- Demo notebook s vizualizací a porovnáním algoritmů

## Přehled algoritmů

| Algoritmus | Celková složitost FPRAS | Klíčová myšlenka |
|------------|-------------------------|------------------|
| Naivní MC | $O\bigl(\tfrac{r}{\|r-1\|}\, \tfrac{n^6}{\varepsilon^2}\bigr)$ | přímá simulace |
| Chatterjee | $O(n^2 \Delta^2 \varepsilon^{-2} (\log n + \log \varepsilon^{-1}))$ | pouze efektivní kroky |
| Goldberg | $O(\Delta^2 \bar{d}\, \varepsilon^{-2} \log(\bar{d}\, \varepsilon^{-1}))$ | + včasné ukončení |

Všechny MC algoritmy používají epsilon-first API: uživatel zadá $(\varepsilon, \delta)$, knihovna odvodí počet vzorků a limit kroků.

## Technologie

- C++23 header-only, CMake, GTest, OpenMP
- pybind11, scikit-build-core

## Reference

- Díaz, Goldberg, Richerby, Serna (2014). *Absorption time of the Moran process.* Random Structures & Algorithms.
- Chatterjee, Ibsen-Jensen, Nowak (2017). *Faster Monte-Carlo algorithms for fixation probability of the Moran process on undirected graphs.*
- Goldberg, Lapinskas, Richerby (2019). *Phase transitions of the Moran process and algorithmic consequences.*
- Lieberman, Hauert, Nowak (2005). *Evolutionary dynamics on graphs.* Nature, 433, 312--316.
