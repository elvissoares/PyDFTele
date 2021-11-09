# PyDFTele
The Density Functional Theory for Electrolyte Solutions

$$ \Omega[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] = F_{id}[\{\rho_i(\boldsymbol{r})\}] + F_{exc}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]+ \sum_i \int_{V} [V_i^{(\text{ext})}(\boldsymbol{r})-\mu_i] \rho_i(\boldsymbol{r}) d \boldsymbol{r}- \int_{\partial V}\sigma(\boldsymbol{r}) \psi(\boldsymbol{r})  d \boldsymbol{r} $$

The excess free-energy is written as
$$F_{exc}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] = F_{hs}[\{\rho_i(\boldsymbol{r})\}]+ F_{Coul}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] + F_{ele-corr}[\{\rho_i(\boldsymbol{r})\}] $$

The electrostatic correlation $F_{ele-corr}$ can be described using different approximations as
- bulk fluid density (BFD) [Kierlik and Rosinberg, Phys.Rev.A 44, 5025 (1991); Y. Rosenfeld, J. Chem. Phys. 98, 8126 (1993)]
- functionalized mean spherical approximation (fMSA) [Roth and Gillespie, J. Phys.: Condens. Matter 28, 244006 (2016)]
