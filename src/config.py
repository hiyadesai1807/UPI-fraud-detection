"""Central configuration: paths, seeds and feature-tier definitions.

Every member imports from here so that all experiments share the same
data, split and evaluation protocol.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "fraud_dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

for _d in (PROCESSED_DIR, MODELS_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
DL_SEEDS = [42, 7, 2024]          # deep models are trained with 3 seeds
TARGET = "is_fraud"
GROUP = "user_id"

# Split proportions (by user, so no user appears in two partitions)
TEST_SIZE = 0.20
VAL_SIZE = 0.20                   # of the full data

# --------------------------------------------------------------------------
# Columns removed before any modelling
# --------------------------------------------------------------------------
# Identifiers / free text: unique per row or scenario-generated strings.
ID_TEXT_COLS = [
    "transaction_id", "user_id", "merchant_id", "ip_address", "device_id",
    "location", "timestamp", "description", "url_referrer",
    "request_description", "request_description_keywords",
]

# Label proxies: values that directly encode the label (Tier 0 -> Tier 1).
#   handle_verification_status : 'unverified' <=> fraud (AUC = 1.000)
#   receiver_account_age       : always 0 for fraud            (AUC = 0.993)
#   receiver_transaction_history: fraud <= 20, legit mean 50   (AUC = 0.947)
LABEL_PROXY_COLS = [
    "handle_verification_status",
    "receiver_account_age",
    "receiver_transaction_history",
]

# Interaction leak found by the pairwise audit (and first noticed through the
# MLP's Integrated-Gradients ranking): for EVERY legitimate row these two columns
# are exactly equal; they differ only in the 'unknown merchant' fraud scenario.
# Each column alone is uninformative (AUC 0.50) so univariate checks miss it.
GEO_PAIR_COLS = ["geographic_disparity", "geographic_location_vs_ip"]

# Behavioural / contextual features whose distributions overlap between the
# classes. This is the "strict" Tier 2 set used for the main model comparison.
BEHAVIOURAL_COLS = [
    "amount",
    "session_duration",
    "transaction_amount_vs_sender_history",
    "transaction_time_of_day",
    "input_timing_consistency",
    "app_switching_frequency",
    "keyboard_input_speed",
    "input_pause_patterns",
    "screen_active_time",
    "background_data_usage",
    "pin_entry_speed",
    "request_amount_roundness",
]

# Leakage-safe user-history features (computed from a user's *earlier*
# transactions only; never from labels). Built in features.py.
HISTORY_COLS = [
    "hist_prior_count",
    "hist_amount_ratio",
    "hist_amount_z",
    "hist_keyboard_dev",
    "hist_pin_dev",
    "hist_timing_dev",
    "hist_secs_since_prev",
    "hist_device_new",
]

# LSTM sequence settings
SEQ_LEN = 5          # current transaction + up to 4 previous ones
