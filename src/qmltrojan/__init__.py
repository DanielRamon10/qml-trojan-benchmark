"""qmltrojan: gate-insertion trojans and triggered backdoors in variational quantum classifiers.

Modules
-------
data         datasets, PCA reduction and angle-encoding scaling
vqc          golden VQC construction, training and (clean) compilation
trojan       malicious transpiler passes (single-gate insertion and triggered backdoor)
io_qasm      OpenQASM 3 export/parse, structural statistics, inventories
sensitivity  output-distribution metrics (TVD, BC, BD) and classification metrics
detect       text-based (BoW/TF-IDF) golden-vs-infected detection with cross-validation
explain      SHAP explanations of what drives trojan impact
plotting     shared figure helpers
"""

__version__ = "0.1.0"
