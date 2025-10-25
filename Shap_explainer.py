import numpy as np
import math
import networkx as nx
from itertools import combinations

class GeoCausalSHAP:
    """
    GeoCausalSHAP integrates causal structure (DAG) with SHAP to provide causally-informed explanations.
    """

    def __init__(self, dag):
        """
        Initializes the GeoCausalSHAP with a causal graph (DAG).

        Parameters:
            dag (networkx.DiGraph or similar): Directed acyclic graph representing causal relationships.
        """
        self.dag = dag

    def explain_instance(self, x, model, background_data, mode="mode 1", location_indices=None):
        """
        Computes SHAP values for a single instance using either mode 1 or mode 2.

        Parameters:
            x (np.ndarray): Feature vector to explain.
            model: Trained predictive model with .predict().
            background_data (np.ndarray): Background dataset for missing feature imputation.
            mode (str): Explanation mode: 'asymmetric' or 'interventional'.
            location_indices (iterable of int, optional): Indices of location features to be fixed.

        Returns:
            np.ndarray: SHAP values for each feature.
        """
        phi = np.zeros_like(x)

        if mode == "mode 1":
            phi = self._causal_shap_1(x, model, background_data, K_max=3)
        elif mode == "mode 2":
            for i in range(len(x)):
                phi[i] = self._causal_shap_2(x, model, i, background_data)
        else:
            raise ValueError("Mode must be 'mode1' or 'mode2'")

        return phi

    def _causal_shap_1(self, x, model, background_data, K_max=100):
        """
        Fast Mode-1 GeoCausal SHAP:
            - batched predictions over background_data
            - memoization of f(x_S)
            - generate only ancestral-closed subsets up to size K_max (default 2)
        """
        # --- set up ---
        x = np.asarray(x)
        B = background_data.shape[0]
        n_features = x.shape[0]
        shap_values = np.zeros(n_features, dtype=float)

        # Sanity: require a DAG
        if not isinstance(self.dag, nx.DiGraph) or not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("GeoCausalSHAP mode-1 requires a directed acyclic graph (DAG).")

        # Precompute ancestors/descendants and a topo order (used for pruning & ordering)
        anc_cache = [set(nx.ancestors(self.dag, i)) for i in range(n_features)]
        desc_cache = [set(nx.descendants(self.dag, i)) for i in range(n_features)]
        topo = list(nx.topological_sort(self.dag))
        topo_pos = {v: i for i, v in enumerate(topo)}

        # Helper: generate ONLY ancestral-closed subsets ⊆ allowed, with |S| ≤ K_max
        def generate_closed_subsets(allowed):
            # process allowed in topological order to maximize pruning
            allowed_sorted = sorted(allowed, key=lambda v: topo_pos[v])
            subsets = [set()]  # start from empty set
            for v in allowed_sorted:
                # we may add v only if all its ancestors are already inside S
                extendable = []
                for S in subsets:
                    if len(S) < K_max and anc_cache[v].issubset(S):
                        extendable.append(S)
                # extend
                for S in extendable:
                    subsets.append(S | {v})
            # keep only those with size ≤ K_max (already enforced), unique by construction
            return subsets

        # Cache for f(x_S) across all i (same background, same x): huge speed-up
        fx_cache = {}

        def fx_of(S):
            """Mean prediction over background with features in S set to x."""
            key = tuple(sorted(S))
            if key in fx_cache:
                return fx_cache[key]
            # build a modified background matrix once, then batch-predict
            M = background_data.copy()
            if S:
                cols = np.fromiter(S, dtype=int)
                M[:, cols] = x[cols]
            val = float(np.mean(model.predict(M)))
            fx_cache[key] = val
            return val

        # --- main loop over target feature i ---
        fact = math.factorial  # local binding
        denom = fact(n_features)  # constant in Shapley weight

        for i in range(n_features):
            # Exclude i and its descendants from candidate set
            disallowed = desc_cache[i] | {i}
            allowed = [j for j in range(n_features) if j not in disallowed]

            # Generate only ancestral-closed subsets up to size K_max
            closed_subsets = generate_closed_subsets(allowed)

            contrib_sum = 0.0
            weight_sum = 0.0

            for S in closed_subsets:
                k = len(S)
                # Shapley weight using full-dim M: k!(M-k-1)! / M!
                w = (fact(k) * fact(n_features - k - 1)) / denom

                fx_S = fx_of(S)
                S_plus_i = frozenset(S | {i})
                fx_Si = fx_of(S_plus_i)

                contrib_sum += (fx_Si - fx_S) * w
                weight_sum += w

            shap_values[i] = (contrib_sum / weight_sum) if weight_sum > 0 else 0.0

        return shap_values

    def _causal_shap_2(self, x, model, i, background_data):
        """
        Exact Mode-2 GeoCausal SHAP for feature i (accelerated):
          - Batched predictions over background_data
          - Memoization of f(x_S)
          - Still enumerates all subsets (no approximation)
        """
        x = np.asarray(x)
        n = x.shape[0]
        other = [j for j in range(n) if j != i]

        # descendants of i (break i->desc paths)
        try:
            descendants_i = nx.descendants(self.dag, i)
        except Exception:
            descendants_i = set()
        D = frozenset(descendants_i)

        # memoized batched prediction
        fx_cache = {}

        def fx_of(S_fro):
            if S_fro in fx_cache:
                return fx_cache[S_fro]
            M = background_data.copy()
            if S_fro:
                cols = np.fromiter(S_fro, dtype=int)
                M[:, cols] = x[cols]
            val = float(np.mean(model.predict(M)))
            fx_cache[S_fro] = val
            return val

        phi_i = 0.0
        n_fact = math.factorial(n)

        # enumerate all subsets of other features
        for k in range(len(other) + 1):
            for subset in combinations(other, k):
                S = frozenset(subset)
                S_base = S - D
                S_i = S_base | frozenset({i})

                fx_S = fx_of(S_base)
                fx_Si = fx_of(S_i)

                w = (math.factorial(k) * math.factorial(n - k - 1)) / n_fact
                phi_i += w * (fx_Si - fx_S)

        return phi_i





