# demo/method_1_bert.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import plotly.graph_objects as go


# ------------------------------------------------------
# Helper: Create Sankey Diagram (Course -> Domain)
# ------------------------------------------------------
def create_sankey_diagram(course_domain_pairs: List[Tuple[str, str]]):
    """
    course_domain_pairs: list of (course_id, domain_name)
    """

    if not course_domain_pairs:
        return go.Figure().update_layout(
            title_text="No course-domain matches above threshold",
            font_size=12
        )

    courses = sorted(list({c for c, _ in course_domain_pairs}))
    domains = sorted(list({d for _, d in course_domain_pairs}))

    labels = courses + domains

    course_to_idx = {c: i for i, c in enumerate(courses)}
    domain_to_idx = {d: i + len(courses) for i, d in enumerate(domains)}

    source = []
    target = []
    value = []

    # Each match contributes weight 1 from course -> domain
    for c, d in course_domain_pairs:
        source.append(course_to_idx[c])
        target.append(domain_to_idx[d])
        value.append(1)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=["#1f77b4"] * len(courses) + ["#ff9900"] * len(domains),
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color="#cccccc",
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Courses in each AACN Domain",
        font_size=12,
        height=900,
    )

    return fig


# ------------------------------------------------------
# Helper: Similarity & Coverage
# ------------------------------------------------------
def compute_similarity_and_coverage(
    embedder: SentenceTransformer,
    syllabi: List[Dict],
    aacn_domains: Dict[str, List[str]],
    threshold: float = 0.50,
):
    """
    Uses course-level texts (objectives/outcomes/outline) and AACN PIs to compute:
      - coverage_report (Markdown)
      - match_df (top matching pairs)
      - matrix_df (course x domain similarity)
      - course_domain_pairs (for Sankey)
    """

    # --------- Build course text units ----------
    course_ids: List[str] = []
    course_texts: List[str] = []

    # we also track which syllabus row each text came from
    owner_index: List[int] = []

    for idx, s in enumerate(syllabi):
        course_id = s.get("course_title") or s.get("source_file", "").split(".")[0] or f"course_{idx}"

        for field in ["learning_objectives", "learning_outcomes", "topical_outline"]:
            for txt in s.get(field, []):
                if txt:
                    course_ids.append(course_id)
                    course_texts.append(txt)
                    owner_index.append(idx)

    if not course_texts or not aacn_domains:
        coverage_report = (
            "### Domain Coverage Report\n"
            "No course text or AACN domains available for similarity analysis."
        )
        empty_df = pd.DataFrame()
        return coverage_report, empty_df, empty_df, []

    # --------- Embed course texts and domain PIs ----------
    course_embs = embedder.encode(course_texts, show_progress_bar=False)

    domain_embs_dict = {
        domain: embedder.encode(pis, show_progress_bar=False)
        for domain, pis in aacn_domains.items()
        if pis
    }

    coverage_flags: Dict[str, bool] = {d: False for d in aacn_domains.keys()}
    rows_for_matrix: Dict[str, Dict[str, float]] = {}
    match_rows: List[Tuple[str, str, str, float]] = []  # (course_id, text, domain, score)
    course_domain_pairs: List[Tuple[str, str]] = []

    # Initialize matrix structure
    for cid in course_ids:
        rows_for_matrix.setdefault(cid, {d: np.nan for d in aacn_domains.keys()})

    # --------- Compute similarities ----------
    for domain, dom_embs in domain_embs_dict.items():
        sim_matrix = cosine_similarity(course_embs, dom_embs)
        max_scores = sim_matrix.max(axis=1)

        for i, score in enumerate(max_scores):
            if score >= threshold:
                cid = course_ids[i]
                txt = course_texts[i]
                score_f = float(score)

                coverage_flags[domain] = True
                match_rows.append((cid, txt, domain, score_f))
                course_domain_pairs.append((cid, domain))

                current = rows_for_matrix[cid].get(domain, np.nan)
                if np.isnan(current) or score_f > current:
                    rows_for_matrix[cid][domain] = score_f

    # --------- Coverage report ----------
    covered_domains = [d for d, flag in coverage_flags.items() if flag]
    uncovered_domains = [d for d, flag in coverage_flags.items() if not flag]

    coverage_report = "### Domain Coverage Report\n"
    coverage_report += "**Covered Domains:**\n"
    if covered_domains:
        for d in covered_domains:
            coverage_report += f"- {d}\n"
    else:
        coverage_report += "- None\n"

    coverage_report += "\n**Not Covered Domains:**\n"
    if uncovered_domains:
        for d in uncovered_domains:
            coverage_report += f"- {d}\n"
    else:
        coverage_report += "- None\n"

    # --------- DataFrames ----------
    if match_rows:
        match_df = pd.DataFrame(
            match_rows, columns=["Course ID", "Course Text", "Domain", "Similarity"]
        ).sort_values("Similarity", ascending=False)
    else:
        match_df = pd.DataFrame(columns=["Course ID", "Course Text", "Domain", "Similarity"])

    # course x domain matrix
    matrix_df = pd.DataFrame.from_dict(rows_for_matrix, orient="index")
    matrix_df.index.name = "Course ID"

    # deduplicate course-domain pairs for Sankey
    course_domain_pairs = sorted(list(set(course_domain_pairs)))

    return coverage_report, match_df, matrix_df, course_domain_pairs


# ------------------------------------------------------
# Main: BERTopic + Similarity + Sankey
# ------------------------------------------------------
def run_bert_topic(
    all_docs: List[str],
    syllabi: List[Dict],
    aacn_domains: Dict[str, List[str]],
    threshold: float = 0.50,
):
    """
    all_docs     : list[str]  – combined text for BERTopic (e.g., course texts + PIs)
    syllabi      : list[dict] – syllabus JSON objects (each with course fields + text fields)
    aacn_domains : dict       – {domain_name: [Progression Indicator strings]}
    threshold    : float      – similarity threshold for domain coverage and tables

    Returns:
      fig_topics        : Plotly Figure
      fig_hierarchy     : Plotly Figure
      fig_heatmap       : Plotly Figure
      coverage_report   : Markdown string
      match_df          : pandas DataFrame (top matches)
      matrix_df         : pandas DataFrame (course x domain similarities)
      sankey_fig        : Plotly Figure (course -> domain)
    """

    # ---------------------------
    # 1. Embeddings + BERTopic
    # ---------------------------
    print("Loading embedding model: multi-qa-mpnet-base-dot-v1...")
    embedder = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

    print("Computing embeddings for BERTopic...")
    embeddings = embedder.encode(all_docs, show_progress_bar=True)

    print("Running BERTopic (UMAP + HDBSCAN clustering)...")
    topic_model = BERTopic(embedding_model=embedder)
    topics, probs = topic_model.fit_transform(all_docs, embeddings)

    fig_topics = topic_model.visualize_topics()
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_heatmap = topic_model.visualize_heatmap()

    # ---------------------------
    # 2. Similarity & coverage
    # ---------------------------
    coverage_report, match_df, matrix_df, course_domain_pairs = compute_similarity_and_coverage(
        embedder=embedder,
        syllabi=syllabi,
        aacn_domains=aacn_domains,
        threshold=threshold,
    )

    # ---------------------------
    # 3. Sankey diagram
    # ---------------------------
    sankey_fig = create_sankey_diagram(course_domain_pairs)

    return (
        fig_topics,
        fig_hierarchy,
        fig_heatmap,
        coverage_report,
        match_df,
        matrix_df,
        sankey_fig,
    )
