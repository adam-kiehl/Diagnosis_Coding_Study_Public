# data_entry_app.py
from __future__ import annotations

# Import packages
import regex
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
from html import escape
from filelock import FileLock
import torch
import numpy as np
import pandas as pd
import streamlit as st
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Import utilities
from data_entry.models import GatorTronClassifier

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
TEST_RECORDS_CSV = APP_DIR / "records_df.csv"  # test records
INDEX_TABLE_CSV = APP_DIR / "index_table.csv"  # list of codes
ALL_CODES_PT = APP_DIR / "all_codes.pt"  # code indices
MODEL_PATH = APP_DIR / "../fine_tuned/model_save_gator.pkl"  # trained model


# Codes with duplicate names to remove
duplicate_codes = [
    83486002,
    111510002,
    93453004,
    90039006,
    39567001,
    204358001,
    25121006,
    118943001,
    267888004,
    92518001,
    44462005,
    3192005,
    399079008,
    92828000,
]


# ---------- Helpers ----------
def load_test_records() -> pd.DataFrame:
    """Import dataframe of test records for coding."""
    # Read test records from .csv
    df = pd.read_csv(TEST_RECORDS_CSV)
    # Normalize 'ai' to boolean if it looks like 0/1 or strings
    if df["ai"].dtype != bool:
        ai_numeric = pd.to_numeric(df["ai"], errors="coerce")
        ai_bool = ai_numeric.fillna(0).astype(int).astype(bool)
        df["ai"] = ai_bool
    return df


def load_diag_codes() -> pd.DataFrame:
    """Import dataframe of available codes."""
    # Read list of codes from .csv
    index_table = pd.read_csv(INDEX_TABLE_CSV)
    # Ensure expected columns exist
    expected_cols = {"concept_code", "concept_id", "concept_name"}
    missing = expected_cols - set(index_table.columns)
    if missing:
        raise ValueError(f"index_table.csv missing columns: {missing}")
    # Ensure concept_code is string type
    index_table["concept_code"] = index_table["concept_code"].astype(str)
    # Merge codes list with index list
    all_codes = pd.DataFrame(
        {
            "concept_code": pd.Series(torch.load(ALL_CODES_PT).numpy()).astype(
                str
            )
        }
    )
    index_table = all_codes.merge(index_table, how="left", on="concept_code")
    return index_table


@st.cache_data()
def load_diag_relationships() -> pd.DataFrame:
    """Import dataframe of concept relationships."""
    # Read concept relationship table
    concept_relationship = pd.read_csv("CONCEPT_RELATIONSHIP.csv", sep="\t")
    return concept_relationship


# NEW: Precompute and cache parent lookup once (Fix #1)
@st.cache_data(show_spinner=False)
def build_parent_lookup(
    index_table: pd.DataFrame, concept_relationship: pd.DataFrame
):
    """Build a dict {child_concept_id: [ {concept_id, concept_code, concept_name}, ... ]}.
    This removes per-row filtering/merge during rendering.
    """
    idx = (
        index_table[["concept_id", "concept_code", "concept_name"]]
        .drop_duplicates()
        .copy()
    )
    idx["concept_id"] = pd.to_numeric(
        idx["concept_id"], errors="coerce"
    ).astype("Int64")
    idx = idx.dropna(subset=["concept_id"]).astype({"concept_id": int})

    # Filter relationships once
    rel = concept_relationship.loc[
        (concept_relationship["relationship_id"] == "Is a")
        & (concept_relationship["invalid_reason"].isna()),
        ["concept_id_1", "concept_id_2"],
    ].copy()

    # Keep only parents that exist in index_table
    valid_parent_ids = set(idx["concept_id"].tolist())
    rel = rel[rel["concept_id_2"].isin(valid_parent_ids)]

    # Attach parent code & name once
    rel = rel.merge(
        idx, left_on="concept_id_2", right_on="concept_id", how="left"
    )

    # Build lookup
    parent_lookup = (
        rel.groupby("concept_id_1")
        .apply(
            lambda g: [
                {
                    "concept_id": int(r["concept_id"]),
                    "concept_code": str(r["concept_code"]),
                    "concept_name": str(r["concept_name"]),
                }
                for _, r in g.iterrows()
            ]
        )
        .to_dict()
    )
    return parent_lookup


def select_random_record(
    df: pd.DataFrame, username: str, use_ai: bool
) -> Optional[pd.Series]:
    """Return random unlabeled record to user."""
    # Enforce valid username
    if not username:
        return None
    # Consider blanks as NaN or empty strings
    user_col = df[username]
    is_blank = user_col.isna() | (user_col.astype(str).str.strip() == "")
    # Find unlabeled records subset
    subset = df[is_blank & (df["ai"] == use_ai)]
    if subset.empty:
        return None
    return subset.sample(1).iloc[0]


def compute_similarity(a: str, b: str) -> float:
    """Compute similarity between two concepts."""
    # Clean terms to compare
    a_norm = (a or "").casefold().strip()
    b_norm = (b or "").casefold().strip()
    if not a_norm or not b_norm:
        return 0.0
    # Basic matching ratio
    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    # Substring boost
    if a_norm in b_norm:
        ratio = max(
            ratio, min(1.0, 0.6 + len(a_norm) / max(len(b_norm), 1) * 0.4)
        )
    return float(ratio)


def search_diag_codes(
    df: pd.DataFrame, query: str, top_k: int = 5
) -> pd.DataFrame:
    """Search list of available codes for matches."""
    # Handle empty search
    if not query or df.empty:
        return df.head(0)
    # Compute similarity scores
    scores = []
    for idx, row in df.iterrows():
        concept = (
            str(row["concept_name"]) if pd.notna(row["concept_name"]) else ""
        )
        score = compute_similarity(query, concept)
        scores.append((idx, score))
    # Determine best matches
    scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in scores[:top_k]]
    return df.loc[top_indices]


# ---------- AI Functions ----------
@st.cache_resource
def get_tokenizer(model_option):
    """Return model-specific tokenizer.

    Args:
        model_option (string): Selected model to use.

    # TODO: Fill in data type
    Returns:
        _type_: Model-specific tokenizer.
    """

    if model_option == "GatorTron":
        tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base")

    return tokenizer


@st.cache_resource
def load_ai_model(dim: int):
    """Load the GatorTron model and tokenizer."""
    try:
        # Initialize GatorTron model
        model = GatorTronClassifier(2560, dim)
        # Load trained model weights
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device("cpu"))[
                "model_state_dict"
            ]
        )
        # Get tokenizer
        tokenizer = get_tokenizer("GatorTron")
        # Set model to evaluation mode
        model.eval()
        return model.to("cuda"), tokenizer
    except Exception as e:
        st.error(f"Error loading AI model: {e}")
        return None, None


def run_ai_prediction(model, tokenizer, text: str):
    """Run AI prediction on clinical text and return raw predictions tensor."""
    try:
        # Handle missing breed token
        if text and text[0] != "[":
            text = "[-1.0] " + text
        # Tokenize input text
        input_ids = tokenizer(
            [text], return_tensors="pt", padding=False, truncation=True
        )["input_ids"].to("cuda")
        # Truncate input text if needed
        if input_ids.size()[1] > 512:
            input_ids = input_ids[:, 0:512]
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids, (input_ids != 0).to(torch.int64))
            predictions = torch.sigmoid(outputs)
        return predictions
    except Exception as e:
        st.error(f"Error running AI prediction: {e}")
        return None


# ---------- HTML Functions ----------
# FIX #3: Rewrote HTML builders to use list-append + ''.join() and use parent_lookup


def create_ai_html(predictions, top_k, index_table, parent_lookup):
    """Generate HTML for AI predicted codes with confidence bars using list-based building."""
    # Clipboard helper
    script_block = (
        "<script>"
        "function copyCode(txt){navigator.clipboard.writeText(txt);}"
        "</script>"
    )
    parts = [
        script_block,
        '<div style="font-family: Arial; max-height:525px; font-size:14px; overflow-y:scroll; padding-right:10px">',
    ]
    # Get the top k values and indices
    top_values, top_indices = torch.topk(predictions, k=top_k)
    # Convert PyTorch tensors to NumPy arrays
    top_values_np = top_values[0].detach().cpu().numpy()
    top_indices_np = top_indices[0].detach().cpu().numpy()
    # Pre-extract arrays for faster iloc
    names = index_table["concept_name"].to_numpy()
    codes = index_table["concept_code"].to_numpy()
    ids = index_table["concept_id"].to_numpy()

    # Loop through top k predicted codes
    for i in range(top_k):
        # Calculate the confidence percentage
        confidence = round(float(top_values_np[i]) * 100, 2)
        # Collect concept name, code, and ID
        idx = int(top_indices_np[i])
        concept_name = escape(str(names[idx]))
        concept_code = escape(str(codes[idx]))
        concept_id = int(ids[idx])

        if int(concept_code) not in duplicate_codes:
            # Add the label and confidence bar to the HTML parts
            parts.append(f"""
    <p><b>{concept_name}</b> (
    <a href=\"https://athena.ohdsi.org/search-terms/terms/{concept_id}\" target=\"_blank\">{concept_code}</a>
    <button onclick=\"copyCode('{concept_code}')\" style=\"margin-left:6px; padding:2px 6px; font-size:12px; max-height:100px;\">Copy</button>
    ) <br>
    """)
            # Parent concepts via precomputed lookup
            parents = parent_lookup.get(concept_id, [])
            for j, p in enumerate(parents):
                p_name = escape(p["concept_name"])
                p_code = escape(p["concept_code"])
                p_id = int(p["concept_id"])
                parts.append(f"""
    <div style=\"white-space: pre-wrap;\">|\t{p_name} (<a href=\"https://athena.ohdsi.org/search-terms/terms/{p_id}\" target=\"_blank\">{p_code}</a> <button onclick=\"copyCode('{p_code}')\" style=\"margin-left:6px; padding:2px 6px; font-size:12px; max-height:100px;\">Copy</button>)</div>
    <br>
    """)
            parts.append(f"""
    <div style=\"width:98%; border:1px solid black\">
        <div style=\"height:15px; width:{confidence}%; background-color:purple;\"></div>
    </div>
    {confidence}%</p>
    """)

    # End the HTML code
    parts.append("</div> </div>")
    return "".join(parts)


def create_search_html(results: pd.DataFrame, parent_lookup):
    # Build the full HTML block including styles, script, and content
    parts = ["""
<style>
.copy-row{ display:flex; align-items:center; padding:6px 0; border-bottom:1px solid #eee; }
.copy-col-concept{ flex:2; font-weight:400; white-space:pre-wrap; }
.copy-col-code{ flex:1; display:flex; align-items:center; gap:8px; }
.copy-col-code .code-text{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.copy-col-code .copy-btn{ margin-left:auto; }
.copy-row.copy-header .copy-col-code{ justify-content:flex-start; }
.copy-header{ font-weight:400; text-transform:capitalize; }
</style>
<script>
function copyTxt(t){navigator.clipboard.writeText(t);}
</script>
<div style=\"max-height:400px; overflow-y:scroll; padding-right:8px;\">
<div class='copy-row copy-header'>
    <div class='copy-col-concept' style='font-weight:600;'>Concept</div>
    <div class='copy-col-code' style='font-weight:600;'>
        <span class='code-text'>Code</span>
    </div>
</div>
"""]
    # Add rows
    for _, r in results.iterrows():
        concept_name = escape(str(r["concept_name"]))
        concept_code = escape(str(r["concept_code"]))
        concept_id = int(r["concept_id"])

        if int(concept_code) not in duplicate_codes:
            row_html = f"""
    <div class='copy-row'>
    <div class='copy-col-concept'><b>{concept_name}</b></div>
    <div class='copy-col-code'>
        <span class='code-text'><a href=\"https://athena.ohdsi.org/search-terms/terms/{concept_id}\" target=\"_blank\">{concept_code}</a></span>
        <button class='copy-btn' onclick=\"copyTxt('{concept_code}')\" style='padding:2px 6px;font-size:12px;'>Copy</button>
    </div>
    </div>
    """
            parts.append(row_html)
            # Append parents via lookup
            parents = parent_lookup.get(concept_id, [])
            for j, p in enumerate(parents):
                p_name = escape(p["concept_name"])
                p_code = escape(p["concept_code"])
                p_id = int(p["concept_id"])
                parts.append(f"""
    <div class='copy-row'>
    <div class='copy-col-concept'>|\t{p_name}</div>
    <div class='copy-col-code'>
        <span class='code-text'><a href=\"https://athena.ohdsi.org/search-terms/terms/{p_id}\" target=\"_blank\">{p_code}</a></span>
        <button class='copy-btn' onclick=\"copyTxt('{p_code}')\" style='padding:2px 6px;font-size:12px;'>Copy</button>
    </div>
    </div>
    """)

    parts.append("</div>")  # Close scroll container

    return "".join(parts)


# ---------- Streamlit App ----------
st.set_page_config(
    page_title="Data Entry Tool for AI Diagnosis Coding Validation",
    page_icon="https://brand.colostate.edu/wp-content/uploads/sites/47/2024/03/CSU-Symbol-r-357-RGB.png",
    layout="wide",
)
st.title("Data Entry Tool for AI Diagnosis Coding Validation")
st.divider()
# Session state initialization
if "selected_record_id" not in st.session_state:
    st.session_state.selected_record_id = None
if "timer_start" not in st.session_state:
    st.session_state.timer_start = None
if "cached_username" not in st.session_state:
    st.session_state.cached_username = ""
if "last_ai_flag" not in st.session_state:
    st.session_state.last_ai_flag = None
if "ai_predictions" not in st.session_state:
    st.session_state.ai_predictions = None

# Load dataframes
try:
    test_records_df = load_test_records()
    index_table = load_diag_codes()
    concept_relationship = load_diag_relationships()
    # Build parent lookup once (Fix #1)
    parent_lookup = build_parent_lookup(index_table, concept_relationship)
except Exception as e:
    st.error(f"Error loading CSVs: {e}")
    st.stop()

# User login inputs
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input(
            "Name:", value=st.session_state.cached_username, autocomplete="off"
        ).lower()
        # Status message: completed and remaining counts for AI and non-AI
        if name.strip():
            # Use existing dataframe without persisting new columns yet
            if name in test_records_df.columns:
                user_series = test_records_df[name]
                is_blank = user_series.isna() | (
                    user_series.astype(str).str.strip() == ""
                )
                ai_mask = test_records_df["ai"] == True
                non_ai_mask = test_records_df["ai"] == False
                total_ai = int(ai_mask.sum())
                total_non_ai = int(non_ai_mask.sum())
                completed_ai = int((~is_blank & ai_mask).sum())
                completed_non_ai = int((~is_blank & non_ai_mask).sum())
                remaining_ai = max(0, total_ai - completed_ai)
                remaining_non_ai = max(0, total_non_ai - completed_non_ai)
                st.caption(
                    f"AI: {completed_ai} done / {remaining_ai} left • Non-AI: {completed_non_ai} done / {remaining_non_ai} left"
                )
            else:
                st.error(
                    f"Column '{name}' not found in records. Please add this user column."
                )
    with col2:
        st.text("Should AI be used for this entry?")
        use_ai = st.checkbox("AI", value=False)
    # CSU Green Hex: #007A33
    csu_green = "#28724F"
    # Inject custom CSS
    st.markdown(
        f"""
<style>
div.stButton > button:first-child {{
  background-color: {csu_green};
  color: white;
  border: none;
  padding: 0.5em 1em;
  border-radius: 5px;
  font-weight: bold;
}}
div.stButton > button:first-child:hover {{
  background-color: {csu_green};
  color: white;
}}
</style>
""",
        unsafe_allow_html=True,
    )
    # Button to get record
    if st.button("Give Me a Record!"):
        if not name.strip():
            st.warning("Please enter your name before requesting a record.")
        else:
            # Flag to clear codes_text and search box on next rerun
            st.session_state.clear_codes_text = True
            st.session_state.clear_search = True

            # Validate required user column exists before selection
            if name not in test_records_df.columns:
                st.error(
                    f"Column '{name}' not found in records. Please add this user column to continue."
                )
                st.stop()

            # Reset record-related states
            st.session_state.selected_record_id = None
            st.session_state.timer_start = None
            st.session_state.ai_predictions = None

            # Select a random record
            record = select_random_record(test_records_df, name, use_ai)
            if record is None:
                st.info(
                    "No available records match your selection. You have completed this section of the experiment!"
                )
                st.session_state.selected_record_id = None
                st.session_state.timer_start = None
                st.session_state.ai_predictions = None
            else:
                st.session_state.selected_record_id = int(record["record_id"])
                st.session_state.timer_start = time.time()
                st.session_state.cached_username = name
                st.session_state.last_ai_flag = bool(use_ai)

                # Run AI prediction if AI checkbox is selected
                if use_ai:
                    with st.spinner("Running AI analysis..."):
                        model, tokenizer = load_ai_model(index_table.shape[0])
                        if model is not None:
                            combined_text = f"{record['diagnosis']} {record['assessment']} {record['pathology']}"
                            predictions = run_ai_prediction(
                                model, tokenizer, combined_text
                            )
                            st.session_state.ai_predictions = predictions
                        else:
                            st.session_state.ai_predictions = None
                else:
                    st.session_state.ai_predictions = None
st.divider()

# Display selected record
record_container = st.container()
with record_container:
    st.subheader("Record")
    if st.session_state.selected_record_id is None:
        st.caption("No record selected yet.")
    else:
        rec = test_records_df.loc[
            test_records_df["record_id"] == st.session_state.selected_record_id
        ]
        if rec.empty:
            st.warning("Previously selected record no longer exists.")
        else:
            row = rec.iloc[0]
            # Display diagnosis only if not blank/NaN
            if pd.notna(row["diagnosis"]) and str(row["diagnosis"]).strip():
                st.markdown("**Diagnosis**")
                st.write(row["diagnosis"])
            # Display assessment only if not blank/NaN
            if pd.notna(row["assessment"]) and str(row["assessment"]).strip():
                st.markdown("**Assessment**")
                st.write(row["assessment"])
            # Display pathology only if not blank/NaN
            if pd.notna(row["pathology"]) and str(row["pathology"]).strip():
                st.markdown("**Pathology**")
                st.write(row["pathology"])
st.divider()

# Side-by-side: AI Results and Search
left_col, right_col = st.columns(2)

# Display AI results
with left_col:
    st.subheader("AI Results")
    if st.session_state.ai_predictions is None:
        st.info(
            "AI-derived results will appear here when AI checkbox is selected."
        )
    else:
        # Generate HTML with confidence bars (using parent_lookup)
        html_confidence = create_ai_html(
            st.session_state.ai_predictions,
            (st.session_state.ai_predictions >= 0.1).sum().item(),
            index_table,
            parent_lookup,
        )
        # Display predicted codes with confidence bars
        st.components.v1.html(html_confidence, height=525)

# Display search results
with right_col:
    st.subheader("Code Search")
    # Search box
    # Initialize session state for search clearing
    if "clear_search" not in st.session_state:
        st.session_state.clear_search = False

    # Clear search box if flagged (before rendering the widget)
    if st.session_state.clear_search:
        st.session_state.search_box = ""  # Clear the widget value
        st.session_state.clear_search = False

    # Render the search box
    query = st.text_input("Search:", key="search_box", autocomplete="off")

    # Perform search only if query is non-empty
    if query.strip():
        results = search_diag_codes(index_table, query, top_k=5)
    else:
        results = index_table.head(0)

    # Display search results
    st.text("Search Results:")
    if results.empty:
        st.caption("No results yet.")
    else:
        with st.container():
            full_html = create_search_html(results, parent_lookup)
            st.components.v1.html(full_html, height=400)

st.divider()

# Codes input and submit
st.subheader("Enter Applicable Codes")
# Initialize session state for codes_text and clear flag
if "codes_text" not in st.session_state:
    st.session_state.codes_text = ""
if "clear_codes_text" not in st.session_state:
    st.session_state.clear_codes_text = False
# Clear codes_text if flagged (before rendering the widget)
if st.session_state.clear_codes_text:
    st.session_state.codes_text = ""
    st.session_state.clear_codes_text = False
# Render the textbox
codes_text = st.text_area(
    "Codes (enter codes separated by commas):", key="codes_text", height=38
)

# Submit button
submit_disabled = (
    st.session_state.selected_record_id is None or not (name or "").strip()
)
if st.button("Submit", disabled=submit_disabled):
    if st.session_state.selected_record_id is None:
        st.warning("No record selected to submit.")
    elif not (name or "").strip():
        st.warning("Please enter your name.")
    else:
        # Reload to avoid stale data before write
        df = load_test_records()

        # Validate required columns exist
        if name not in df.columns:
            st.error(
                f"Column '{name}' not found in records. Please add this user column before submitting."
            )
            st.stop()

        match_mask = df["record_id"] == st.session_state.selected_record_id
        if not match_mask.any():
            st.error("Could not find the selected record to update.")
        else:
            # Ensure the user column is string type
            df[name] = df[name].astype(str)

            # Save codes entered by user
            df.loc[match_mask, name] = (
                str(st.session_state.codes_text)
                .replace(" ", "")
                .replace(",", "\n")
            )

            # Time tracking
            start = st.session_state.timer_start
            elapsed = None
            if isinstance(start, (int, float)):
                elapsed = max(0.0, time.time() - float(start))

            # Write to dynamic time column per user
            time_col = f"time_{name}"
            if time_col not in df.columns:
                st.error(
                    f"Column '{time_col}' not found in records. Please add this time column before submitting."
                )
                st.stop()

            df.loc[match_mask, time_col] = elapsed

            # Write to order column
            order_col = f"order_{name}"
            if order_col not in df.columns:
                st.error(
                    f"Column '{order_col}' not found in records. Please add this time column before submitting."
                )
                st.stop()

            is_blank = (
                df[name].isna()
                | (df[name].astype(str).str.strip() == "")
                | (df[name].astype(str).str.strip() == "nan")
            )
            df.loc[match_mask, order_col] = int((~is_blank).sum())

            lock_path = TEST_RECORDS_CSV.with_suffix(
                TEST_RECORDS_CSV.suffix + ".lock"
            )

            # Write updated records dataframe to .csv
            try:
                with FileLock(lock_path, timeout=10):
                    df.to_csv(TEST_RECORDS_CSV, index=False)

                # Reset selection and timer
                st.session_state.selected_record_id = None
                st.session_state.timer_start = None

                if elapsed is not None:
                    st.success(
                        f"Submission saved. Time: {elapsed:.1f} seconds."
                    )
                else:
                    st.success("Submission saved.")

                # Flag to clear codes_text on next rerun
                st.session_state.clear_codes_text = True

            except TimeoutError:
                st.error(
                    "Could not acquire file lock. Another user may be writing. Please try again shortly."
                )
                st.stop()

st.divider()

# Instructions Display Box
st.subheader("Instructions")
html_code = """
<style>
 .instruction-box { border: 1px solid #ddd; padding: 20px; border-radius: 8px; background-color: #f0f2f6; margin-bottom: 20px; list-style-position: inside; }
 .instruction-box h4 { margin-top: 0; }
</style>
<div class=\"instruction-box\">
 <ol>
 <li>Enter your name in the \"Name\" textbox and press enter.</li>
 <li>Select whether AI inference should be used for this entry using the \"AI\" textbox.</em></li>
 <li>Click \"Give Me a Record!\" to continue. A random record should be displayed.</li>
 <li>If the \"AI\" checkbox is selected, AI-derived results will be displayed below. If no AI results are displayed, no predictions with sufficient confidence were generated.</li>
 <li>Use the AI-derived results and/or the code search tool to identify diagnosis codes that should be applied to the displayed record.</li>
 <li>Parent codes from the SNOMED hierarchy will be displayed beneath primary codes.</li>
 <li>Copy and paste relevant codes into the \"Enter Applicable Codes\" textbox separated by commas (spaces are fine).</li>
 <li>Click \"Submit\" to save your entered codes.</li>
 <li>Retrieve your next record by clicking \"Give Me a Record!\" again.</li>
 </ol>
</div>
"""
st.html(html_code)
