# qml-trojan-benchmark

**Security of Quantum Machine Learning against trojan / backdoor insertion.**

A reproducible, laptop-scale benchmark that studies what happens when a compromised quantum
software-supply-chain component (a malicious Qiskit transpiler pass) inserts gates into an
already-trained **Variational Quantum Classifier (VQC)**. It bridges two research directions:
**(1) trojan detection in quantum circuits** and **(2) Quantum Adversarial Machine Learning (QAML)**.

The study asks three questions:

- **(a)** What is the behavioural and **classification-accuracy** impact of single-gate insertions in VQCs?
- **(b)** Can a **trigger** degrade/flip classification only for specific inputs while preserving the rest (a **backdoor**)?
- **(c)** Can simple **text classifiers** (BoW/TF-IDF + classical ML) over the exported QASM tell clean VQCs from infected ones?

> This repository deliberately imitates the methodology of the quantum-trojan sensitivity work
> (gate-insertion dataset, TVD/BC/BD metrics, violin/strip plots, a text-detection table) and adds
> QAML-specific metrics (clean accuracy, accuracy drop, and attack success rate for a triggered backdoor).

---

## Adversarial model (software supply chain)

A once-trusted optimization dependency — a Qiskit **transpiler pass** — is compromised and begins
**inserting operations during compilation** of an already-trained VQC. The attacker controls
**neither** the algorithm source, **nor** the cloud provider, **nor** the hardware. The **defender**
has access to the **golden** (clean) and **infected** compiled artifacts and to their measurements.

This is implemented literally: the attack is a real `qiskit.transpiler.TransformationPass` run inside
a `PassManager`, so the infected circuit is produced by the compilation flow itself — exactly as in
the field.

---

## Method at a glance

| Stage | What it does |
|---|---|
| **Victim models** | Train golden VQCs: `{Z, ZZ}` angle encoding + `RealAmplitudes` ansatz, parity-expectation readout, over Iris / digits 0-vs-1 / Breast-Cancer (PCA to 2 or 4 qubits), 3 seeds. |
| **Trojan insertion** | A malicious pass inserts **one** gate `{H,T,X,Y,Z}` per variant in `random` / `idle` / `controlled` mode; 15 variants per model with full structural metadata. |
| **Triggered backdoor** | A conditional trojan (inverse-encoding detector + mid-circuit measured ancilla + classically controlled parity flip) that flips the class only under a trigger pattern. |
| **Sensitivity** | TVD / BC / BD over measured output distributions (512 shots × 15 reps) **vs a golden-vs-golden sampling baseline**, plus clean accuracy and accuracy drop; SHAP over structural attributes. |
| **Detection** | QASM-as-text (BoW & TF-IDF, uni+bigrams) × {LogReg, MultinomialNB, ComplementNB, LinearSVC} × {GroupKFold, StratifiedKFold}; full metric suite + confusion matrices. |

The victim/model architecture, the metrics, and every design decision are explained in depth, in
Portuguese, in [`docs/explicacao.md`](docs/explicacao.md).

---

## Installation

Requires **Python ≥ 3.10** (developed on 3.12), CPU only.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the `qmltrojan` package
```

All dependency versions are pinned in [`requirements.txt`](requirements.txt).

---

## Reproducing the results

```bash
# Validate the whole pipeline end to end in a few minutes:
python scripts/run_all.py --quick

# Full benchmark (~1–1.5 h on a laptop CPU); writes to results/full/:
python scripts/run_all.py --config full
```

Run the stages individually with the same `--quick` / `--config full` flag:

```bash
python scripts/exp1_golden.py     --config full   # train golden VQCs, export QASM, clean accuracy
python scripts/exp2_trojans.py    --config full   # infect, sensitivity (TVD/BC/BD), accuracy drop, SHAP
python scripts/exp3_backdoor.py   --config full   # triggered backdoor: ASR vs clean accuracy
python scripts/exp4_detection.py  --config full   # text detection grid + confusion matrices
```

Run the tests and the didactic notebook:

```bash
pytest -m "not slow"          # fast unit tests
pytest -m slow                # end-to-end quick-pipeline test (a few minutes)
jupyter notebook notebooks/demo.ipynb
```

Outputs land in `results/<mode>/`: `golden/`, `infected/`, `backdoor/` (OpenQASM 3 files),
CSVs (`metadata`, `sensitivity`, `backdoor`, `detection_summary`, `inventory`, `errors`, `shap_*`),
and `figures/*.png`. Quick-mode outputs are git-ignored; full-mode outputs are versioned.

---

## Why OpenQASM 3?

A trained VQC is still a **parametric program**: weights are fixed numbers but data features are
run-time inputs. OpenQASM 2 cannot represent free parameters (`QASM2ExportError`), so each classifier
is exported as **OpenQASM 3** with features declared as `input float[64] _x_i_;`. Export/parse
round-trips are tested.

---

## Results

<!-- RESULTS:START -->
*(Filled in automatically after the full run — see commit history.)*
<!-- RESULTS:END -->

---

## Honest discussion, limitations and negative results

- **Detection is modest, not solved.** Treating QASM as a bag of tokens captures *lexical* traces of
  an inserted gate but **not the circuit's semantics**. Under `GroupKFold` (the realistic setting,
  where the detector is tested on unseen base circuits) performance is clearly weaker than under the
  optimistic `StratifiedKFold`. We report recall **together with** balanced accuracy, F1 and ROC-AUC
  precisely so that a trivial "flag-everything" detector cannot look good.
- **Diagonal gates can be invisible to the label.** `Z` and `T` inserted right before a `Z`-basis
  measurement commute with it and leave the class unchanged, even though they are "trojans"; their
  impact appears only when inserted earlier (before the ansatz). The benchmark shows both.
- **The backdoor works, but with caveats.** On the product `z` encoding the conditional trigger is
  exact (high ASR with small clean-accuracy leakage, most stealthy at larger trigger width `k`). On
  the entangling `zz` encoding the per-qubit trigger is only approximate and the attack **degrades** —
  we measure and report this rather than overstating effectiveness.
- **Classical simulation, few qubits, small datasets.** Everything runs on a noiseless simulator with
  ≤ 6 qubits and PCA-reduced datasets; results need not transfer to noisy hardware or larger models.
- **We do not re-apply full optimization flows.** The clean flow uses `optimization_level=0`; we do
  not study how aggressive transpiler optimization would interact with (or cancel) an inserted gate.

---

## Future work

- **Structural / semantic circuit representations** for detection (DAG features, graph neural
  networks) instead of text tokens.
- Evaluate **after different transpilation flows and optimization levels**, and on **noisy backends**.
- Backdoors keyed to the **entangling (`zz`) encoding** with exact (entanglement-aware) triggers.
- Larger qubit counts and native datasets (no PCA).

---

## References

The data below were confirmed from the source PDFs. Entries marked **[VERIFICAR]** could not be
independently confirmed and should be checked before citing.

1. Hayashi, V. T., & Ruggiero, W. V. (2025). *Hardware Trojan Detection in Open-Source Hardware
   Designs Using Machine Learning.* IEEE Access, 13, 37771–37788. doi:10.1109/ACCESS.2025.3546156
2. Souza Filho, W. J. de, Hayashi, V. T., & Ferreira, B. K. (2026). *Dataset e Análise de
   Sensibilidade de Trojans em Circuitos Quânticos.* Anais Estendidos do 26º SBSeg (WTICG), 489–499.
3. Petenazzi, L. F., & Hayashi, V. T. (2026). *Distribuição de Chaves Quânticas com BB84: simulação e
   análise do QBER com Qiskit.* REIC, 24(1), 612–620. doi:10.5753/reic.2026.8121
4. Das, S., & Ghosh, S. (2023). *TrojanNet: Detecting trojans in quantum circuits using machine
   learning.* arXiv:2306.16701.
5. Das, S., & Ghosh, S. (2024). *Trojan taxonomy in quantum computing.* IEEE ISVLSI 2024, 644–649.
6. John, J., Golla, L., & Wang, Q. (2025). *Stealthy conditional trojans in quantum circuits.*
   ISQED 2025, 1–7.

*Other QAML references (e.g. adversarial robustness of quantum classifiers) are intentionally omitted
here because their bibliographic data could not be confirmed — add them marked **[VERIFICAR]**.*

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Resumo em português

**Contexto.** Este repositório estuda a **segurança de classificadores quânticos variacionais (VQCs)**
contra a inserção de **trojans/backdoors** por um componente comprometido da cadeia de suprimentos de
software quântico — concretamente, um **passe de transpilação malicioso** do Qiskit.

**Modelo adversarial.** Uma dependência de compilação antes confiável passa a **inserir operações
durante a compilação** de um VQC já treinado; o atacante não controla o código-fonte, o provedor ou o
hardware, e o defensor tem acesso às versões **limpa (golden)** e **infectada** e às suas medições.

**Pergunta de pesquisa e metodologia.** (a) Qual o impacto de inserções de porta única sobre o
comportamento e a **acurácia** do VQC? (b) É possível um **backdoor com gatilho** que inverta a
classificação só para entradas específicas? (c) Um **detector textual** (BoW/TF-IDF sobre o QASM)
distingue limpos de infectados? Treinamos VQCs golden (codificação `Z`/`ZZ` + ansatz `RealAmplitudes`,
leitura por expectativa de paridade), inserimos trojans via um `TransformationPass` real, medimos
**TVD/BC/BD** (512 shots × 15 repetições) **contra uma linha de base golden×golden**, além de acurácia
e **ASR**, e avaliamos a detecção com **GroupKFold** e **StratifiedKFold**.

**Resultados principais** (ver a seção *Results* e as figuras em `results/full/figures/`): portas
**não-diagonais** (`X`, `Y`) têm o maior impacto na distribuição e na acurácia, enquanto `Z`/`T`
diagonais podem ser inócuas junto à medição; o **backdoor** atinge **ASR alta** mantendo a acurácia
limpa na codificação `z` (mais furtivo com gatilho mais largo), mas **degrada** na codificação `zz` —
reportado **honestamente**; a **detecção textual é modesta**, especialmente sob `GroupKFold`.

**Limitações.** Simulação clássica sem ruído, poucos qubits, datasets pequenos (com PCA), detecção
textual que **não modela a semântica** do circuito e sem reaplicação de fluxos completos de
otimização. **Trabalhos futuros:** representações **estruturais/semânticas** (DAG/grafos), avaliação
após diferentes fluxos de transpilação e em **backends com ruído**, e gatilhos exatos para a
codificação emaranhada.

**Documentação detalhada e decisões de projeto:** [`docs/explicacao.md`](docs/explicacao.md).
