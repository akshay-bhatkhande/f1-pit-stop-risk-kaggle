"""Strategy-desk triage UI — rank cars for near-term pit attention."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `from src...` works when Streamlit is
# launched from the repo root or from app/ (streamlit sets cwd variously).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data.load import SOURCE_KAGGLE, data_source_label, infer_data_source
from src.eval.metrics import recall_at_topk

REPO_ROOT = _REPO_ROOT
DEMO_DIR = REPO_ROOT / "models" / "demo"
BUNDLE_PATH = DEMO_DIR / "triage_bundle.joblib"
META_PATH = DEMO_DIR / "meta.json"
OOF_PARQUET = DEMO_DIR / "oof_eval.parquet"
OOF_CSV = DEMO_DIR / "oof_eval.csv"
REPORTS = REPO_ROOT / "docs" / "reports"


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def load_oof() -> pd.DataFrame | None:
    if OOF_PARQUET.exists():
        return pd.read_parquet(OOF_PARQUET)
    if OOF_CSV.exists():
        return pd.read_csv(OOF_CSV)
    return None


def resolve_provenance() -> dict:
    meta: dict = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
        except json.JSONDecodeError:
            meta = {}
    source = meta.get("data_source") or infer_data_source()
    return {
        "data_source": source,
        "label": meta.get("data_source_label") or data_source_label(source),
        "n_rows": meta.get("n_rows_trained"),
        "official": bool(meta.get("metrics_are_official_kaggle"))
        or source == SOURCE_KAGGLE,
    }


def main() -> None:
    st.set_page_config(
        page_title="F1 Pit-Stop Risk Lab",
        page_icon=None,
        layout="wide",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
        html, body, [class*="css"]  {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .brand {
            font-family: 'Barlow Condensed', sans-serif;
            font-size: 2.6rem;
            letter-spacing: 0.04em;
            color: #f2f2f2;
            margin-bottom: 0.1rem;
        }
        .tagline {
            color: #b8c0cc;
            font-size: 1.05rem;
            margin-bottom: 1.2rem;
        }
        .hero {
            background: radial-gradient(1200px 400px at 10% -10%, #3a1020 0%, transparent 55%),
                        linear-gradient(135deg, #12151c 0%, #1a2230 45%, #0e1218 100%);
            border: 1px solid #2a3344;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }
        .caveat {
            background: #1a1f28;
            border-left: 3px solid #c45c26;
            padding: 0.85rem 1rem;
            color: #d7dde6;
            font-size: 0.92rem;
        }
        .synth-banner {
            background: #3a1808;
            border: 1px solid #c45c26;
            color: #ffd7b8;
            padding: 0.75rem 1rem;
            margin-bottom: 0.85rem;
            font-size: 0.95rem;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <div class="brand">F1 PIT-STOP RISK LAB</div>
          <div class="tagline">Ops triage under imbalance — flag cars for near-term pit attention.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    provenance = resolve_provenance()
    if not provenance["official"]:
        n_note = (
            f" · trained on {provenance['n_rows']:,} rows"
            if provenance.get("n_rows")
            else ""
        )
        st.markdown(
            f"""
            <div class="synth-banner">
              DATA SOURCE: {provenance["label"]}{n_note}.
              Metrics and scores below are <strong>not</strong> official Kaggle Playground S6E5 results.
              To use real CSVs: place <code>~/.kaggle/kaggle.json</code>, run
              <code>FORCE=1 bash scripts/download_data.sh</code>, then
              <code>python -m src.models.train</code> and reload.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.success(f"Data source: {provenance['label']}")

    bundle = load_bundle()
    oof = load_oof()

    if bundle is None or oof is None:
        st.error(
            "Model artifact not found. From the repo root run:\n\n"
            "1. `bash scripts/download_data.sh` **or** `python -m src.data.generate_demo`\n"
            "2. `python -m src.models.train`\n"
            "3. Reload this app."
        )
        st.stop()

    primary = bundle.get("primary", "hist_gbm")
    st.caption(
        f"Primary model: **{primary}** · held-out OOF scores for triage controls · "
        f"source: **{provenance['label']}**"
    )

    col_a, col_b, col_c = st.columns([1.1, 1, 1])
    with col_a:
        mode = st.radio(
            "Alert budget control",
            ["Top-k % of laps", "Probability threshold"],
            horizontal=True,
        )
    n = len(oof)
    y_true = oof["y_true"].to_numpy()
    y_score = oof["y_score"].to_numpy()

    if mode == "Top-k % of laps":
        with col_b:
            k_pct = st.slider("Review budget (top % of eval laps)", 1, 25, 10)
        k_frac = k_pct / 100.0
        k = max(1, int(np.floor(k_frac * n)))
        thresh = float(np.partition(y_score, -k)[-k]) if k < n else float(y_score.min())
        mask = y_score >= thresh
        # stabilize ties by taking exact top-k
        order = np.argsort(-y_score)[:k]
        mask = np.zeros(n, dtype=bool)
        mask[order] = True
        recall = recall_at_topk(y_true, y_score, k_frac)
        alert_count = int(mask.sum())
    else:
        with col_b:
            thresh = st.slider("Probability threshold", 0.05, 0.95, 0.35, 0.01)
        mask = y_score >= thresh
        alert_count = int(mask.sum())
        k_frac = alert_count / n if n else 0.0
        recall = (
            float(y_true[mask].sum() / y_true.sum()) if y_true.sum() > 0 else 0.0
        )

    with col_c:
        st.metric("Alerts (false-alarm load proxy)", f"{alert_count:,}")
        st.metric("Recall @ current budget", f"{recall:.1%}")

    st.subheader("Attention list")
    show_cols = [
        c
        for c in [
            "Race",
            "Driver",
            "LapNumber",
            "Position",
            "Stint",
            "Compound",
            "TyreLife",
            "RaceProgress",
            "TrackStatus",
            "y_score",
            "y_true",
        ]
        if c in oof.columns
    ]
    ranked = oof.loc[mask, show_cols].sort_values("y_score", ascending=False)
    ranked = ranked.rename(columns={"y_score": "P(pit next)", "y_true": "actual_pit"})
    st.dataframe(ranked.head(250), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Score distribution (OOF)")
        hist_df = pd.DataFrame({"score": y_score, "pit": y_true})
        st.bar_chart(hist_df.groupby(pd.cut(hist_df["score"], bins=20), observed=True)["pit"].mean())

    with right:
        st.subheader("Segment notes")
        slice_files = sorted(REPORTS.glob("slice_*.csv")) if REPORTS.exists() else []
        if slice_files:
            pick = st.selectbox("Slice table", [p.name for p in slice_files])
            st.dataframe(pd.read_csv(REPORTS / pick), use_container_width=True, hide_index=True)
        else:
            st.info("Run training to populate docs/reports/slice_*.csv")

    caveat_data = (
        "Shipped metrics are on <em>synthetic demo</em> CSVs (~22k laps) unless you "
        "downloaded official Kaggle files and retrained."
        if not provenance["official"]
        else "Trained on official Kaggle Playground S6E5 CSVs (local CV — not a medal claim)."
    )
    st.markdown(
        f"""
        <div class="caveat">
          <strong>Caveats</strong><br/>
          {caveat_data}
          Playground data is inspired by an F1 strategy dataset — treat this as ML literacy
          + ops framing, not race-engineering truth. Not live telemetry.
          Watch late-stint and caution-regime slices for higher error.
          Accuracy is not the hero metric under ~4:1 imbalance; use PR-AUC and recall@top-k.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
