import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from notears.linear import notears_linear
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import log, inf
from itertools import combinations


def bic_score(y_true, y_pred, n_params, n_samples):
    # BIC = n * log(MSE) + k * log(n)
    mse = mean_squared_error(y_true, y_pred)
    return n_samples * log(mse + 1e-8) + n_params * log(n_samples)


def custom_ges(X: pd.DataFrame, max_parents=3):
    n_samples, n_features = X.shape
    G = nx.DiGraph()
    G.add_nodes_from(range(n_features))

    current_parents = {i: [] for i in range(n_features)}
    current_scores = {}

    # Step 1: Initialize with empty parent set BICs
    for i in range(n_features):
        y = X.iloc[:, i].values
        y_pred = np.full_like(y, y.mean())
        score = bic_score(y, y_pred, n_params=1, n_samples=n_samples)
        current_scores[i] = score

    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_edge = None

        # Try adding edges i -> j (i ≠ j)
        for j in range(n_features):
            if len(current_parents[j]) >= max_parents:
                continue
            for i in range(n_features):
                if i == j or i in current_parents[j]:
                    continue
                trial_parents = current_parents[j] + [i]
                X_parents = X.iloc[:, trial_parents].values
                y = X.iloc[:, j].values

                try:
                    model = LinearRegression().fit(X_parents, y)
                    y_pred = model.predict(X_parents)
                    new_score = bic_score(y, y_pred, n_params=len(trial_parents), n_samples=n_samples)
                except Exception:
                    continue

                delta = current_scores[j] - new_score  # Positive delta means improvement
                if delta > best_delta:
                    best_delta = delta
                    best_edge = (i, j)
                    best_new_score = new_score

        # Apply best edge addition if improvement found
        if best_edge:
            i, j = best_edge
            current_parents[j].append(i)
            current_scores[j] = best_new_score
            G.add_edge(i, j)
            improved = True

    return G


def _as_array(X):
    if isinstance(X, pd.DataFrame):
        return X.values
    return np.asarray(X)


def _bic_linear(y, Z):
    """
    BIC for linear Gaussian regression y ~ Z (+ intercept).
    Uses: BIC = n*log(RSS/n) + k*log(n), where k includes intercept.
    """
    y = np.asarray(y).reshape(-1, 1)
    n = y.shape[0]
    if Z is None or Z.size == 0:
        yhat = np.full_like(y, y.mean())
        rss = float(((y - yhat) ** 2).sum())
        k = 1  # intercept only
    else:
        Z = np.asarray(Z)
        Z_ = np.hstack([np.ones((n, 1)), Z])  # add intercept
        beta, _, _, _ = np.linalg.lstsq(Z_, y, rcond=None)
        yhat = Z_.dot(beta)
        rss = float(((y - yhat) ** 2).sum())
        k = Z_.shape[1]  # intercept + parents
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + k * np.log(n)


def _local_bic_for_nodes(G, X, nodes):
    """
    Sum BIC over the given nodes only, using current parent sets in G.
    """
    Xarr = _as_array(X)
    score = 0.0
    for j in nodes:
        parents = list(G.predecessors(j))
        Z = Xarr[:, parents] if parents else None
        y = Xarr[:, j]
        score += _bic_linear(y, Z)
    return score


def orient_cpdag_to_dag_smart(G_in, X, prefer_parents=None):
    """
    Convert CPDAG (e.g., GES output) into DAG by orienting mutual edges using local BIC,
    and always avoiding cycles by discarding high-BIC or ambiguous edges.
    """
    if not isinstance(G_in, nx.DiGraph):
        raise ValueError("Expected a DiGraph as input.")

    nodes = list(G_in.nodes())
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)

    # --- Step 1: Classify edges ---
    mutual = set()
    single = set()

    for u, v in G_in.edges():
        if G_in.has_edge(v, u):
            mutual.add(tuple(sorted((u, v))))
        else:
            single.add((u, v))

    # --- Step 2: Add singly directed edges (only if no cycle) ---
    for u, v in single:
        dag.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(dag):
            dag.remove_edge(u, v)  # drop if forms a cycle

    Xarr = _as_array(X)
    prefer_parents = set(prefer_parents) if prefer_parents else set()

    # --- Step 3: Resolve mutual edges smartly ---
    for u, v in mutual:
        dag_uv = dag.copy()
        dag_uv.add_edge(u, v)
        cyc_uv = not nx.is_directed_acyclic_graph(dag_uv)
        bic_uv = _local_bic_for_nodes(dag_uv, Xarr, nodes=[u, v]) if not cyc_uv else np.inf

        dag_vu = dag.copy()
        dag_vu.add_edge(v, u)
        cyc_vu = not nx.is_directed_acyclic_graph(dag_vu)
        bic_vu = _local_bic_for_nodes(dag_vu, Xarr, nodes=[u, v]) if not cyc_vu else np.inf

        # Apply prior bias
        if u in prefer_parents and v not in prefer_parents:
            bic_uv -= 1e-6
        if v in prefer_parents and u not in prefer_parents:
            bic_vu -= 1e-6

        # Select direction (always ensuring DAG)
        if np.isinf(bic_uv) and np.isinf(bic_vu):
            continue  # both directions form cycle → drop
        elif not cyc_uv and (bic_uv < bic_vu or cyc_vu):
            dag.add_edge(u, v)
        elif not cyc_vu and (bic_vu < bic_uv or cyc_uv):
            dag.add_edge(v, u)
        else:
            # both directions OK and same score → pick lower BIC
            if bic_uv <= bic_vu:
                dag.add_edge(u, v)
            else:
                dag.add_edge(v, u)

    # Final check
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Final DAG still has a cycle, which should not happen.")

    return dag

def learn_causal_graph(X, method="notears"):
    if method == "notears":
        W = notears_linear(X.values, lambda1=0.1, loss_type='l2')
        G = nx.DiGraph(W.T)
        print("Adjacency matrix (float weights):")
        print(np.round(W, 3))
        print("\nAdjacency matrix (edges):")
        print((np.abs(W) > 1e-5).astype(int))
        print("NOTEARS (smart-oriented DAG) edges:", G.number_of_edges())
        return G

    elif method == "pc":
        cg = pc(data=X.values, indep_test=fisherz, alpha=0.3, stable=True)
        G_raw = convert_to_nx_graph(cg)  # DiGraph with possible mutual edges
        # Optional: choose prior parents (e.g., exogenous/spatial variables)

        prior = None  # or set([...]) with node ids or names matching G_raw
        G = orient_cpdag_to_dag_smart(G_raw, X, prefer_parents=prior)
        print("PC (smart-oriented DAG) edges:", G.number_of_edges())
        return G

    elif method == "ges":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        print("Running GES with default BIC scoring function...")
        G_raw = custom_ges(X_scaled)  # returns DiGraph, may contain mutual edges
        prior = None  # or set([...]) as above
        G = orient_cpdag_to_dag_smart(G_raw, X_scaled, prefer_parents=prior)
        print("GES (smart-oriented DAG) edges:", G.number_of_edges())
        return G

    else:
        raise ValueError("Unsupported method. Choose from ['notears', 'pc', 'ges'].")


# turn the output of causal-learn:Graph into networkx.DiGraph
def convert_to_nx_graph(cg):
    g_obj = cg['G'] if isinstance(cg, dict) else cg.G

    # create directed graph from adjacency matrix directly
    adj_matrix = g_obj.graph.astype(int)
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return G


def visualize_dag(G, feature_names=None):
    pos = nx.spring_layout(G, seed=42)  # fix the layout
    plt.figure(figsize=(10, 7))

    labels = {i: name for i, name in enumerate(feature_names)} if feature_names else None
    nx.draw(G, pos, with_labels=True, labels=labels,
            node_color='lightblue', node_size=2000,
            arrows=True, font_size=10, edge_color='gray')

    plt.title("Learned Causal Graph (DAG)")
    plt.tight_layout()
    plt.show()
