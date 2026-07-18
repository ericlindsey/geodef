# GeoDef glossary and notation

This page is the source of truth for the terms and symbols used in GeoDef's
documentation and tutorial course. Coordinate, sign, unit, ordering, and
regularization rules are specified in [conventions](conventions.md).

## Core inverse-problem notation

| Term | Symbol | GeoDef name | Meaning |
|---|---:|---|---|
| Data vector | $d$ | `dataset.obs` | Observed GNSS components, InSAR line of sight, or vertical motion. |
| Model vector | $m$ | `result.slip_vector` | Slip parameters in the selected basis; the expert view is blocked by component. |
| Green's matrix | $G$ | `geodef.greens.matrix(...)` | Linear map from unit slip on each patch to predicted observations. |
| Prediction | $Gm$ | `result.predicted`; `invert.prediction(...)` | Noise-free data implied by the estimated model. |
| Residual | $r=d-Gm$ | `result.residuals`; `invert.residual(...)` | Observed minus predicted data. |
| Noise | $\epsilon$ | synthetic noise; covariance metadata | Measurement and model discrepancy added to the prediction. |
| Data covariance | $C_d$ | `dataset.covariance` | Variances and correlations assumed for the data errors. |
| Weight/precision matrix | $W=C_d^{-1}$ | `geodef.greens.stack_weights(...)` | Inverse covariance used in weighted misfit. |
| Regularization operator | $L$ | `regularization=` | Linear operator that encodes smoothing, damping, stress, or a custom prior. |
| Regularization strength | $\lambda$ | `regularization_strength=` | Weight multiplying $\lVert L(m-m_{ref})\rVert^2$; it is not squared again. |
| Reference model | $m_{ref}$ | `regularization_target=` | Model toward which regularization pulls the solution. |
| Objective | $\Phi(m)$ | solved by `geodef.solve(...)` | Weighted data misfit plus regularization penalty. |
| Model covariance | $C_m$ | `geodef.invert.model_covariance(...)` | Linear-Gaussian covariance conditional on the assumed geometry, noise, and regularization. |
| Model resolution | $R$ | `geodef.invert.model_resolution(...)` | Map from true model parameters to their expected recovered values. |
| Chi-squared | $\chi^2$ | `diagnostic.chi2` | Unreduced weighted squared residual, $r^TWr$. |
| Reduced chi-squared | $\chi^2_\nu$ | `result.reduced_chi2` | Chi-squared divided by degrees of freedom. |
| Root-mean-square residual | RMS | `diagnostic.rms` | Unweighted residual scale in the data's declared units. |
| Weighted RMS | WRMS | `diagnostic.wrms` | Residual scale accounting for observation weights. |

## Faults and slip

| Term | Symbol | GeoDef name | Meaning |
|---|---:|---|---|
| Fault patch | — | one element of `Fault` | Rectangular or triangular piece with spatially constant slip. |
| Strike | $\phi$ | `fault.strike` | Clockwise angle from north; the fault dips to the strike's right. |
| Dip | $\delta$ | `fault.dip` | Downward angle from horizontal. |
| Rake | $\rho$ | `result.slip_rake`; `rake=` | Slip direction within the fault plane. |
| Strike slip | $s_s$ | `result.strike_slip` | Slip parallel to patch strike under GeoDef's sign convention. |
| Dip slip | $s_d$ | `result.dip_slip` | Slip down/up the fault plane under GeoDef's rake convention. |
| Slip magnitude | $s$ | `result.slip_magnitude` | $\sqrt{s_s^2+s_d^2}$ for each patch. |
| Slip azimuth | — | `slip_azimuth=` | Geographic horizontal direction, clockwise from north. |
| Plate-motion basis | — | `components="plate"` | Rake-parallel and rake-perpendicular coordinates tied to plate motion. |
| Backslip | — | plate-basis amplitude opposite plate motion | Kinematic slip used to represent interseismic locking. |
| Coupling | $c$ | backslip rate / plate rate | Dimensionless locked fraction between zero and one. |
| Scalar seismic moment | $M_0$ | `fault.moment(...)` | $\mu\sum_k A_k s_k$, in N m. |
| Moment magnitude | $M_w$ | `fault.magnitude(...)` | Logarithmic magnitude derived from scalar moment. |
| Shear modulus | $\mu$ | `ElasticMedium.mu` | Elastic rigidity in Pa. |
| Poisson's ratio | $\nu$ | `ElasticMedium.nu` | Dimensionless elastic compressibility parameter. |

## Data and observation geometry

| Term | GeoDef name | Meaning |
|---|---|---|
| GNSS | `geodef.data.gnss(...)` | East, north, and optionally up displacement or velocity at stations. |
| InSAR | `geodef.data.insar(...)` | Surface motion projected onto a radar line-of-sight unit vector. |
| Look vector | `look_e`, `look_n`, `look_u` | Unit vector that defines positive InSAR line of sight in ENU coordinates. |
| Line of sight | LOS | Scalar projection $u_E l_E+u_N l_N+u_U l_U$. |
| Local frame | `geodef.LocalFrame` | WGS84 origin that gives local East, North, Up coordinates physical meaning. |
| Dataset name | `dataset.dataset_name` | Stable label used to split joint predictions and diagnostics. |
| Spatial covariance | `geodef.data.spatial_covariance(...)` | Dense covariance generated from a distance-decay model plus nugget. |

## Inference and assessment

| Term | GeoDef name | Meaning |
|---|---|---|
| Weighted least squares | `method="wls"` | Minimizes covariance-weighted data misfit. |
| Tikhonov regularization | `regularization=...` | Stabilizes an ill-conditioned inverse problem with a quadratic penalty. |
| Laplacian smoothing | `regularization="laplacian"` | Penalizes spatial curvature in slip. |
| Damping | `regularization="damping"` | Penalizes model magnitude using an identity operator. |
| L-curve | `geodef.invert.lcurve(...)` | Selects a trade-off near the corner of misfit versus model norm. |
| ABIC | `geodef.invert.abic_curve(...)` | Evidence-based comparison of regularization strengths. |
| Cross-validation | `regularization_strength="cv"` | Selects strength by held-out prediction error. |
| Bound | `bounds=(lower, upper)` | Per-parameter admissible interval. |
| Linear inequality | `constraints=(C, d)` | Constraint of the form $Cm\le d$. |
| Checkerboard test | tutorial 08 | Synthetic recovery test for a spatially alternating slip pattern. |
| Spike test | tutorial 08 | Synthetic recovery test for one localized nonzero patch. |
| Geometry parameters | $\theta$ | named mapping or expert `theta` array | Nonlinear location, depth, strike, dip, length, and width parameters. |
| Variable projection | `geodef.invert.geometry_search(...)` | Outer nonlinear geometry search with slip solved linearly at each trial. |
| Prior | $p(m)$, $p(\theta)$ | `geodef.bayes` configuration | Probability model for parameters before conditioning on these data. |
| Likelihood | $p(d\mid m,\theta)$ | posterior `log_likelihood` | Data probability under the forward and noise models. |
| Posterior | $p(m,\theta\mid d)$ | `geodef.bayes` result | Parameter probability after combining prior and likelihood. |
| Credible interval | posterior quantiles | Interval containing a stated fraction of conditional posterior probability. |
| R-hat | `result.rhat` | Between/within-chain MCMC convergence diagnostic; values near one are desirable. |
| Effective sample size | ESS; `result.ess` | Approximate number of independent draws represented by correlated samples. |
| Posterior predictive check | `posterior.predict(...)` | Comparison of observed data with replicated or noise-free posterior predictions. |

## Words that are easy to misuse

- **Resolution is not uncertainty.** Resolution describes spatial averaging;
  uncertainty describes spread under an assumed probability model.
- **Regularization is an assumption, not extra data.** A stable, smooth answer
  can still be biased by the operator or strength chosen.
- **Precision is inverse covariance, not decimal precision.** Backend float
  precision is a separate computational setting.
- **A good fit is not validation.** Residuals can be small under incorrect
  geometry, covariance, elastic structure, or nuisance assumptions.
