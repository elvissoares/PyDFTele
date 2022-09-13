# PyDFTele
The Density Functional Theory for Electrolyte Solutions

For an electrolyte solution close to a charged surface with temperature <img src="https://latex.codecogs.com/svg.image?\inline&space;T" title="\inline T" />, total volume <img src="https://latex.codecogs.com/svg.image?\inline&space;V" title="\inline V" />, and chemical potential of each species <img src="https://latex.codecogs.com/svg.image?\inline&space;\mu_i" title="\inline \mu_i" /> specified, the grand potential, <img src="https://latex.codecogs.com/svg.image?\inline&space;\Omega" title="\inline \Omega" />, is written as

$$\Omega[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] = F^\text{id}[\{\rho_i(\boldsymbol{r})\}] + F^\text{exc}[\{\rho_i(\boldsymbol{r})\}]+ F^\text{coul}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]+ \sum_i \int_{V} d \boldsymbol{r} [V_i^{(\text{ext})}(\boldsymbol{r})-\mu_i] \rho_i(\boldsymbol{r})+ \int_{\partial V}d \boldsymbol{r} \sigma(\boldsymbol{r}) \psi(\boldsymbol{r})$$

where $\rho_i(\boldsymbol{r})$ is the local density of the component i, $\psi(\boldsymbol{r})$ is the local electrostatic potential, and $V^\text{ext}_{i}$ is the external potential. The volume of the system is V and $\partial V$ is the boundary of the system. 

The ideal-gas contribution $F^\text{id}$ is given by the exact expression

$$F^\text{id}[\{\rho_i (\boldsymbol{r})\}] = k_B T\sum_i \int_{V} d\boldsymbol{r}\ \rho_i(\boldsymbol{r})[\ln(\rho_i (\boldsymbol{r})\Lambda_i^3)-1]$$

where <img src="https://latex.codecogs.com/svg.image?\inline&space;k_B" title="\inline k_B" /> is the Boltzmann constant, <img src="https://latex.codecogs.com/svg.image?\inline&space;T" title="\inline T" /> is the absolute temperature, and <img src="https://latex.codecogs.com/svg.image?\inline&space;\Lambda_i" title="\inline \Lambda_i" /> is the well-known thermal de Broglie wavelength of each ion.

The Coulomb's free-energy <img src="https://latex.codecogs.com/svg.image?F^\text{coul}" title="F^\text{coul}" /> is obtained by the addition of the electric field energy density and the minimal-coupling of the interaction between the electrostatic potential <img src="https://latex.codecogs.com/svg.image?\psi(\boldsymbol{r})" title="\psi(\boldsymbol{r})" /> and the charge density <img src="https://latex.codecogs.com/svg.image?\inline&space;\sum_i&space;Z_i&space;e\rho_i(\boldsymbol{r})" title="\inline \sum_i Z_i e\rho_i(\boldsymbol{r})" />, and it can be written as 

$$F^\text{coul}[\{\rho_{i}(\boldsymbol{r})\},\psi(\boldsymbol{r})] = -\int_V d\boldsymbol{r}\ \frac{\epsilon_0 \epsilon_r}{2} |\nabla{\psi(\boldsymbol{r})}|^2 + \int_{V} d\boldsymbol{r}\ \sum_i Z_i e \rho_{i}(\boldsymbol{r}) \psi(\boldsymbol{r})$$

where <img src="https://latex.codecogs.com/svg.image?\inline&space;Z_i" title="\inline Z_i" /> is the valence of the ion i, <img src="https://latex.codecogs.com/svg.image?e" title="e" /> is the elementary charge, <img src="https://latex.codecogs.com/svg.image?\epsilon_0" title="\epsilon_0" /> is the vacuum permittivity, and <img src="https://latex.codecogs.com/svg.image?\epsilon_r" title="\epsilon_r" /> is the relative permittivity.

The excess Helmholtz free-energy, <img src="https://latex.codecogs.com/svg.image?\inline&space;F^\text{exc}" title="\inline F^\text{exc}" />, is the free-energy functional due to particle-particle interactions splitted in the form

$$F^\text{exc}[\{\rho_i(\boldsymbol{r})\}] = F^\text{hs}[\{\rho_i(\boldsymbol{r})\}] + F^\text{ec}[\{\rho_i(\boldsymbol{r})\}]$$

where $F^{\textrm{hs}}$ is the hard-sphere excess contribution and $F^{\textrm{ec}}$ is the electrostatic correlation excess contribution. 

The hard-sphere contribution, $F^{\textrm{hs}}$, represents the hard-sphere exclusion volume correlation and it can be described using different formulations of the fundamental measure theory (FMT) as

- [x] **R**osenfeld **F**unctional (**RF**) - [Rosenfeld, Y., Phys. Rev. Lett. 63, 980–983 (1989)](https://link.aps.org/doi/10.1103/PhysRevLett.63.980)
- [x] **W**hite **B**ear version **I** (**WBI**) - [Yu, Y.-X. & Wu, J., J. Chem. Phys. 117, 10156–10164 (2002)](http://aip.scitation.org/doi/10.1063/1.1520530); [Roth, R., Evans, R., Lang, A. & Kahl, G., J. Phys. Condens. Matter 14, 12063–12078 (2002)](https://iopscience.iop.org/article/10.1088/0953-8984/14/46/313)
- [x] **W**hite **B**ear version **II** (**WBII**) - [Hansen-Goos, H. & Roth, R. J., Phys. Condens. Matter 18, 8413–8425 (2006)](https://iopscience.iop.org/article/10.1088/0953-8984/18/37/002)

The electrostatic correlation <a href="https://latex.codecogs.com/gif.latex?F%5E%5Ctext%7Bec%7D" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F%5E%5Ctext%7Bec%7D" title="F^{\textrm{ec}}" /></a> can be described using different approximations as
- [x] **M**ean-**F**ield **T**heory (**MFT**) - <img src="https://latex.codecogs.com/svg.image?\inline&space;F^{\textrm{ec}}&space;=&space;0" title="\inline F^{\textrm{ec}} = 0" />
- [x] **B**ulk **F**luid **D**ensity (**BFD**) - [Kierlik and Rosinberg, Phys.Rev.A 44, 5025 (1991)](https://doi.org/10.1103/PhysRevA.44.5025); [Y. Rosenfeld, J. Chem. Phys. 98, 8126 (1993)](https://doi.org/10.1063/1.464569)
- [x] **f**unctionalized **M**ean **S**pherical **A**pproximation (**fMSA**) - [Roth and Gillespie, J. Phys.: Condens. Matter 28, 244006 (2016)](https://doi.org/10.1088/0953-8984/28/24/244006)
- [ ] **R**eference **F**luid **D**ensity (**RFD**) - [Gillespie, D., Nonner, W. & Eisenberg, R. S., J. Phys. Condens. Matter 14, 12129–12145 (2002)](https://iopscience.iop.org/article/10.1088/0953-8984/14/46/317); [Gillespie, D., Valiskó, M. & Boda, D., J. Phys. Condens. Matter 17, 6609–6626 (2005)](https://iopscience.iop.org/article/10.1088/0953-8984/17/42/002)

Finally, The chemical potential for each ionic species is defined as $\mu_i = \mu_i^\text{id} + \mu_i^\text{exc}$, where superscripts id and exc refer to ideal and excess contributions, respectively.

The thermodynamic equilibrium is obtained by the minimum of the grand-potential, $\Omega$, which can be obtained by the functional derivatives, such that, the equilibrium condition for each charged component is given by 

$$\left. \frac{\delta \Omega}{\delta \rho_i(\boldsymbol{r})} \right\|_{\{\mu_k\},V,T} = k_B T \ln[\Lambda_i^3 \rho_i(\boldsymbol{r})] + \frac{\delta F^\text{exc}}{\delta \rho_i(\boldsymbol{r})} + Z_i e \psi(\boldsymbol{r}) + V^{\text{ext}}_i(\boldsymbol{r}) - \mu_i =0$$

and for the electrostatic potential it is 

$$\left. \frac{\delta \Omega}{\delta \psi(\boldsymbol{r})} \right\|_{\{\mu_k\},V,T} = \epsilon_0 \epsilon_r\nabla^2{\psi(\boldsymbol{r})} + \sum_i Z_i e \rho_i(\boldsymbol{r}) =0$$

valid in the whole volume V, this is the well-known Poisson's equation of the electrostatic potential with the boundary conditions

$$\left. \frac{\delta \Omega}{\delta \psi(\boldsymbol{r})} \right\|_{\{\mu_k\},V,T} = \left. \epsilon_0 \epsilon_r \boldsymbol{\hat{n}}(\boldsymbol{r}) \cdot \boldsymbol{\nabla}{\psi(\boldsymbol{r})} \right\|_{\partial V} + \sigma(\boldsymbol{r}) = 0$$

valid on the boundary surface $\partial V$, where $\boldsymbol{\hat{n}}(\boldsymbol{r})$ is denoting the vector normal to the surface pointing inward to the system.

# Examples

## Voukadinova
|![Figure1](https://github.com/elvissoares/PyDFTele/blob/main/examples/ionprofile-electrolyte-Voukadinova2018-Fig5-Z%2B%3D1-rho%2B%3D0.01M.png)|![Figure2](https://github.com/elvissoares/PyDFTele/blob/main/examples/potential-electrolyte-Voukadinova2018-Fig5-Z%2B%3D1-rho%2B%3D0.01M.png)|
|:--:|:--:|
| <b>Fig.1 - The ionic density profiles of an 1:1 electrolyte solution with c_+= 0.01 M and σ = -0.5C/m². </b>| <b>Fig.2 - The electrostatic potential profile of an 1:1 electrolyte solution with c_+= 0.01 M and σ = -0.5C/m². </b>|
|![Figure3](https://github.com/elvissoares/PyDFTele/blob/main/examples/ionprofile-electrolyte-Voukadinova2018-Fig5-Z%2B%3D1-rho%2B%3D1.0M.png)|![Figure4](https://github.com/elvissoares/PyDFTele/blob/main/examples/potential-electrolyte-Voukadinova2018-Fig5-Z%2B%3D1-rho%2B%3D1.0M.png)|
|:--:|:--:|
| <b>Fig.3 - The ionic density profiles of an 1:1 electrolyte solution with c_+= 1.0 M and σ = -0.5C/m². </b>| <b>Fig.4 - The electrostatic potential profile of an 1:1 electrolyte solution with c_+= 1.0 M and σ = -0.5C/m². </b>|
