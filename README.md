# Quantum-Machine-Learning-with-Qiskit-2.x
Contains jupyter notebooks on various qml algorithms implemented using qiskit 2.x

## Notebooks

| Notebook | Topic | What it covers |
|---|---|---|
| [QML1_Intro_to_Primitives_q2x.ipynb](QML1_Intro_to_Primitives_q2x.ipynb) | Primitives: Sampler & Estimator | Runs simple and parametric circuits as Primitive Unified Blocks (PUBs) with `SamplerV2`/`EstimatorV2` on the Aer simulator, on real IBM Quantum hardware via `QiskitRuntimeService`, and with the local `StatevectorSampler`/`StatevectorEstimator`. Covers transpiling for simulator vs. hardware, mapping observables (`SparsePauliOp`) to a transpiled qubit layout, and reading counts/expectation values/std-devs from job results. |
| [QML2_DataEncoding_FeatureMaps_q2x.ipynb](QML2_DataEncoding_FeatureMaps_q2x.ipynb) | Data encoding / feature maps | Reference notebook of classical-to-quantum encoding schemes: basis, amplitude, angle, phase, and dense-angle encoding, followed by the parameterized circuit families used as feature maps/ansätze — `EfficientSU2`, `ZFeatureMap`, `ZZFeatureMap`, `PauliFeatureMap`, `TwoLocal`, and `NLocal`. Each section has the math definition plus the matching Qiskit snippet. |
| [QML3a_QVC_Toy_Dataset_1parameter.ipynb](QML3a_QVC_Toy_Dataset_1parameter.ipynb) | Variational Quantum Classifier (minimal example) | Builds a 1-qubit, 1-feature/1-weight VQC (`f_θ(x) = ⟨0\|U†(x)W†(θ)OW(θ)U(x)\|0⟩`) on random ±1-labeled toy data. Defines the forward pass, MSE loss, and trains the ansatz with `scipy.optimize.minimize` using `StatevectorEstimator`, then reports train/test accuracy. Smallest, easiest-to-follow VQC example in the series. |
| [QML3b_QVC_Leukemia_q2x.ipynb](QML3b_QVC_Leukemia_q2x.ipynb) | VQC on real data (Leukemia, raw features) | Same VQC architecture as 3a, scaled up to the Golub et al. leukemia gene-expression dataset (~72 patients, 10 selected features) using a `ZZFeatureMap` + `EfficientSU2` ansatz. Trains in batches over epochs with `StatevectorEstimator` and evaluates train/test accuracy. Feature values are used as loaded, without rescaling. |
| [QML3bb_QVC_Leukemia_q2x.ipynb](QML3bb_QVC_Leukemia_q2x.ipynb) | VQC on real data (Leukemia, rescaled features) | Near-duplicate of 3b with one key change: features are rescaled to the [-π, π] range before encoding, which matters for angle/phase-based feature maps since encoded values should stay within a period. Otherwise the same feature-map + ansatz + training/evaluation pipeline. |
| [QML4a_Quantum_SVM_Toy_Dataset_q2x.ipynb](QML4a_Quantum_SVM_Toy_Dataset_q2x.ipynb) | Quantum kernels & Quantum SVM (toy data) | Introduces quantum kernels: computing a single kernel-matrix entry via the inversion test (`unitary_overlap`), then building a full kernel matrix over a toy graph dataset (128 points, 14 features). Feeds the precomputed kernel into scikit-learn's `SVC` and compares accuracy against a classical (RBF) SVM on the same data. Also shows transpiling/optimizing the overlap circuit for real IBM hardware. |
| [QML4b_Quantum_SVM_for_Leukemia_q2x.ipynb](QML4b_Quantum_SVM_for_Leukemia_q2x.ipynb) | Quantum SVM on Leukemia data (custom feature map) | Applies the quantum-kernel SVM approach from 4a to the leukemia dataset using a custom parametric feature map, rescaling features to [0, 2π]. Builds train/test kernel matrices, trains `sklearn.svm.SVC` with the precomputed kernel, visualizes the kernel matrices, and compares against a classical SVM baseline. |
| [QML4c_Quantum_SVM_for_Leukemia_q2x.ipynb](QML4c_Quantum_SVM_for_Leukemia_q2x.ipynb) | Quantum SVM on Leukemia data (ZZFeatureMap) | Same quantum-kernel SVM pipeline as 4b on the leukemia dataset, but swaps the custom feature map for Qiskit's built-in `ZZFeatureMap` + `EfficientSU2` combo when building the kernel circuit. Useful for comparing the built-in feature map against the hand-written one in 4b. |

Notebooks build on each other roughly in numeric order: **1** (primitives) → **2** (encoding schemes) → **3a/3b/3bb** (variational classifiers) → **4a/4b/4c** (quantum-kernel SVMs). All notebooks target qiskit 2.3 syntax; see each notebook's setup cell for the exact `pip install` list (`qiskit`, `qiskit-aer`, `qiskit-ibm-runtime`, `matplotlib`, `pylatexenc`, and for 3/4-series also `pandas`, `scikit-learn`). Course materials/credit: M. Faryad ([github/muf148](https://github.com/muf148)), based on and adapted from IBM Quantum Learning.

## Data Encoding Notes

Notes on the encoding schemes from `QML2_DataEncoding_FeatureMaps_q2x.ipynb` — how classical data $\vec{x} = (x_1,\dots,x_N)$ gets mapped to a quantum state.

### 1. Basis Encoding
Each feature's binary representation is mapped one-to-one onto a block of qubit basis states. For $\vec{x}=(5,7,0)$ with 4-bit features:
$$
5 \to 0101,\quad 7 \to 0111,\quad 0 \to 0000 \;\Rightarrow\; |\vec{x}\rangle = |0101\,0111\,0000\rangle
$$
- Qubits needed: $4N$ for $N$ features at 4-bit precision — scales linearly and expensively with both feature count and precision.
- Lossless, purely classical-style encoding; not typically used to gain any quantum advantage on its own.

### 2. Amplitude Encoding
Each feature becomes an amplitude of the statevector. For $\vec{x}^{(1)}=(4,8,5)$:
$$
\sum_i \left|x^{(1)}_i\right|^2 = 4^2+8^2+5^2 = 105 = |\alpha|^2 \;\Rightarrow\; \alpha=\sqrt{105}
$$
$$
|\psi(\vec{x}^{(1)})\rangle = \frac{1}{\sqrt{105}}\big(4|00\rangle+8|01\rangle+5|10\rangle+0|11\rangle\big)
$$
- Qubits needed: $n$ such that $2^n \ge N$ (pad with zeros if $N$ is not a power of 2) — i.e. $n \approx \log_2(N)$, the most qubit-efficient scheme.
- Implemented in Qiskit via `QuantumCircuit.initialize(desired_state)`.
- Tradeoff: efficient state preparation is not free — arbitrary-state prep can require deep circuits in general.

### 3. Angle Encoding
One feature per qubit, encoded as a rotation angle (commonly $R_Y$), leaving qubits in a product state:
$$
|\vec{x}^{(j)}\rangle = \bigotimes_{k=1}^{N} R_Y\!\big(2\vec{x}^{(j)}_k\big)|0\rangle = \bigotimes_{k=1}^{N} \cos(\vec{x}^{(j)}_k)|0\rangle + \sin(\vec{x}^{(j)}_k)|1\rangle
$$
- Qubits needed: $n \le N$, typically $n = N$ (one qubit per feature).
- Constant circuit depth (depth 1 before transpilation) — cheap on NISQ hardware.
- Data must be rescaled into $(0, 2\pi]$ before encoding to avoid periodic aliasing/information loss (this is exactly why `QML3bb` rescales features to $[-\pi,\pi]$ before training, while `QML3b` does not).

### 4. Phase Encoding
Feature maps to the *phase* of a qubit already placed on the equator by a Hadamard:
$$
|\vec{x}^{(j)}\rangle = \bigotimes_{k=1}^{N} P_k\!\big(\phi=\vec{x}^{(j)}_k\big)|+\rangle^{\otimes N} = \frac{1}{\sqrt{2^N}}\bigotimes_{k=1}^{N}\Big(|0\rangle + e^{i\vec{x}^{(j)}_k}|1\rangle\Big)
$$
- Circuit depth 2 (Hadamard layer + phase-gate layer) — efficient and simple.

### 5. Dense Angle Encoding (DAE)
Combines angle ($R_Y$, polar angle $\theta$) and phase ($R_Z$, azimuthal angle $\phi$) rotations to pack **two** features per qubit:
$$
|\vec{x}\rangle = \bigotimes_{k=1}^{N/2} \cos(x_{2k-1})|0\rangle + e^{i x_{2k}}\sin(x_{2k-1})|1\rangle
$$
- Qubits needed: $N/2$ — halves the qubit count vs. plain angle encoding for the same feature count.
- Generalizes to "general qubit encoding" if the sinusoidal functions are replaced by arbitrary functions of the two features.

### 6–11. Parameterized Feature Maps (used as encoders *and* trainable ansätze)
| Scheme | Idea | Qiskit |
|---|---|---|
| **Efficient SU2** | Alternating single-qubit rotation + entangling layers; a 2-qubit, 1-rep circuit already spans 8 real parameters — much denser encoding than angle/phase encoding per qubit. | `efficient_su2(num_qubits, reps, ...)` |
| **Z Feature Map (ZFM)** | Extension of phase encoding: alternating Hadamard and phase-gate layers, $\mathscr{U}_{\text{ZFM}} = \big(\bigotimes_k P(\vec{x}_k)\big) H^{\otimes N}$, repeated $r$ times. No entanglement — still classically simulable. | built explicitly with `.h()` / `.p()`, or `z_feature_map` |
| **ZZ Feature Map (ZZFM)** | ZFM plus two-qubit $R_{ZZ}(\theta)$ entangling gates between qubit pairs, $\theta_{q,p}=2(\pi-\vec{x}_q)(\pi-\vec{x}_p)$ by default. Conjectured to be classically hard to simulate — this is where a quantum kernel can, in principle, beat a classical one. | `zz_feature_map(feature_dimension, entanglement, reps)` |
| **Pauli Feature Map (PFM)** | Generalizes ZFM/ZZFM to arbitrary Pauli-string exponentials: $U(\vec{x}) = \exp\!\big(i\alpha \sum_{S} \phi_S(\vec{x}) \prod_{i\in S}\sigma_i\big)$, $\sigma_i\in\{I,X,Y,Z\}$, $\alpha=2$ by default (tunable to better align the kernel to data). | `pauli_feature_map(feature_dimension, entanglement, reps)` |
| **TwoLocal** | Generic alternating rotation-layer / entanglement-layer ansatz; not data-specific, mainly used as the trainable $W(\theta)$ block. | `TwoLocal(num_qubits, rotation_blocks, entanglement_blocks, reps)` |
| **NLocal** | Generalization of TwoLocal where rotation/entanglement blocks can act on any number of qubits (not just 1 or 2), giving more flexible circuit topologies. | `NLocal(...)`, custom blocks via `RYGate`/`RZGate`/`CXGate` |

### Rule of thumb
- **Fewer qubits** → amplitude encoding (best) or dense angle encoding, at the cost of harder/deeper state preparation.
- **Shallow, NISQ-friendly circuits** → angle or phase encoding (depth 1–2), one qubit per feature.
- **Potential quantum kernel advantage** → ZZ/Pauli feature maps, which add entanglement that's conjectured hard to simulate classically (used as the feature maps in `QML3b/3bb` and `QML4c`).
- **Always rescale inputs** into the periodic range the encoding expects (e.g. $(0,2\pi]$ or $[-\pi,\pi]$) before angle/phase-based encoding — see the `QML3b` vs `QML3bb` and `QML4b`/`QML4c` rescaling steps.
