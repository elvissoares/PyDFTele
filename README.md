# PyDFTele
The Density Functional Theory for Electrolyte Solutions

<!---
$$ \Omega[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] = F_{id}[\{\rho_i(\boldsymbol{r})\}] + F_{exc}[\{\rho_i(\boldsymbol{r})\}]+ F_{coul}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] + \sum_i \int_{V} [V_i^{(\text{ext})}(\boldsymbol{r})-\mu_i] \rho_i(\boldsymbol{r}) d \boldsymbol{r}+ \int_{\partial V}\sigma(\boldsymbol{r}) \psi(\boldsymbol{r})  d \boldsymbol{r} $$)
-->

<img src="https://latex.codecogs.com/svg.image?\begin{align*}\Omega[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]&space;=\&space;&&space;F^\text{id}[\{\rho_i(\boldsymbol{r})\}]&space;&plus;&space;F^\text{exc}[\{\rho_i(\boldsymbol{r})\}]&plus;&space;F^\text{coul}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]&space;\\&space;&&space;&plus;&space;\sum_i&space;\int_{V}&space;d&space;\boldsymbol{r}&space;&space;[V_i^{(\text{ext})}(\boldsymbol{r})-\mu_i]&space;\rho_i(\boldsymbol{r})&plus;&space;\int_{\partial&space;V}d&space;\boldsymbol{r}&space;\sigma(\boldsymbol{r})&space;\psi(\boldsymbol{r})\end{align*}" title="\begin{align*}\Omega[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] =\ & F^\text{id}[\{\rho_i(\boldsymbol{r})\}] + F^\text{exc}[\{\rho_i(\boldsymbol{r})\}]+ F^\text{coul}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] \\ & + \sum_i \int_{V} d \boldsymbol{r} [V_i^{(\text{ext})}(\boldsymbol{r})-\mu_i] \rho_i(\boldsymbol{r})+ \int_{\partial V}d \boldsymbol{r} \sigma(\boldsymbol{r}) \psi(\boldsymbol{r})\end{align*}" />

The ideal-gas contribution <img src="https://latex.codecogs.com/svg.image?\inline&space;F^\text{id}" title="\inline F^\text{id}" /> is given by the exact expression

<img src="https://latex.codecogs.com/svg.image?F^\text{id}[\{\rho_i&space;(\boldsymbol{r})\}]&space;=&space;k_B&space;T\sum_i&space;\int_{V}&space;d\boldsymbol{r}\&space;\rho_i(\boldsymbol{r})[\ln(\rho_i&space;(\boldsymbol{r})\Lambda_i^3)-1]" title="F^\text{id}[\{\rho_i (\boldsymbol{r})\}] = k_B T\sum_i \int_{V} d\boldsymbol{r}\ \rho_i(\boldsymbol{r})[\ln(\rho_i (\boldsymbol{r})\Lambda_i^3)-1]" />

where <img src="https://latex.codecogs.com/svg.image?\inline&space;k_B" title="\inline k_B" /> is the Boltzmann constant, <img src="https://latex.codecogs.com/svg.image?\inline&space;T" title="\inline T" /> is the absolute temperature, and <img src="https://latex.codecogs.com/svg.image?\inline&space;\Lambda_i" title="\inline \Lambda_i" /> is the well-known thermal de Broglie wavelength of each ion.

The Coulomb's free-energy <img src="https://latex.codecogs.com/svg.image?F^\text{coul}" title="F^\text{coul}" /> is obtained by the addition of the electric field energy density and the minimal-coupling of the interaction between the electrostatic potential <img src="https://latex.codecogs.com/svg.image?\psi(\boldsymbol{r})" title="\psi(\boldsymbol{r})" /> and the charge density <img src="https://latex.codecogs.com/svg.image?\inline&space;\sum_i&space;Z_i&space;e\rho_i(\boldsymbol{r})" title="\inline \sum_i Z_i e\rho_i(\boldsymbol{r})" />, and it can be written as 

<img src="https://latex.codecogs.com/svg.image?F^\text{coul}[\{\rho_{i}(\boldsymbol{r})\},\psi(\boldsymbol{r})]&space;=&space;-\int_V&space;d\boldsymbol{r}\&space;\frac{\epsilon_0&space;\epsilon_r}{2}&space;|\nabla{\psi(\boldsymbol{r})}|^2&space;&plus;&space;\int_{V}&space;d\boldsymbol{r}\&space;\sum_i&space;Z_i&space;e&space;\rho_{i}(\boldsymbol{r})&space;\psi(\boldsymbol{r})" title="F^\text{coul}[\{\rho_{i}(\boldsymbol{r})\},\psi(\boldsymbol{r})] = -\int_V d\boldsymbol{r}\ \frac{\epsilon_0 \epsilon_r}{2} |\nabla{\psi(\boldsymbol{r})}|^2 + \int_{V} d\boldsymbol{r}\ \sum_i Z_i e \rho_{i}(\boldsymbol{r}) \psi(\boldsymbol{r})" />

where <img src="https://latex.codecogs.com/svg.image?\inline&space;Z_i" title="\inline Z_i" /> is the valence of the ion i, <img src="https://latex.codecogs.com/svg.image?e" title="e" /> is the elementary charge, <img src="https://latex.codecogs.com/svg.image?\epsilon_0" title="\epsilon_0" /> is the vacuum permittivity, and <img src="https://latex.codecogs.com/svg.image?\epsilon_r" title="\epsilon_r" /> is the relative permittivity.

The excess free-energy is written as

<img src="https://latex.codecogs.com/svg.image?F^\text{exc}[\{\rho_i(\boldsymbol{r})\}]&space;=&space;F^\text{hs}[\{\rho_i(\boldsymbol{r})\}]&space;&plus;&space;F^\text{ec}[\{\rho_i(\boldsymbol{r})\}]" title="F^\text{exc}[\{\rho_i(\boldsymbol{r})\}] = F^\text{hs}[\{\rho_i(\boldsymbol{r})\}] + F^\text{ec}[\{\rho_i(\boldsymbol{r})\}]" />

where <img src="https://latex.codecogs.com/svg.image?\inline&space;F^{\textrm{hs}}" title="\inline F^{\textrm{hs}}" /> is the hard-sphere excess contribution and <img src="https://latex.codecogs.com/svg.image?\inline&space;F^{\textrm{ec}}" title="\inline F^{\textrm{ec}}" /> is the electrostatic correlation excess contribution. 

The electrostatic correlation <a href="https://latex.codecogs.com/gif.latex?F%5E%5Ctext%7Bec%7D" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F%5E%5Ctext%7Bec%7D" title="F^{\textrm{ec}}" /></a> can be described using different approximations as
- **M**ean-**F**ield **T**heory (**MFT**) - <img src="https://latex.codecogs.com/svg.image?\inline&space;F^{\textrm{ec}}&space;=&space;0" title="\inline F^{\textrm{ec}} = 0" />
- **B**ulk **F**luid **D**ensity (**BFD**) - [Kierlik and Rosinberg, Phys.Rev.A 44, 5025 (1991)](https://doi.org/10.1103/PhysRevA.44.5025); [Y. Rosenfeld, J. Chem. Phys. 98, 8126 (1993)](https://doi.org/10.1063/1.464569)
- **f**unctionalized **M**ean **S**pherical **A**pproximation (**fMSA**) - [Roth and Gillespie, J. Phys.: Condens. Matter 28, 244006 (2016)](https://doi.org/10.1088/0953-8984/28/24/244006)
