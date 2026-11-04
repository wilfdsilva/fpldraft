import json
import datetime
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
import streamlit as st
import concurrent.futures

# --- Page Configuration ---
st.set_page_config(page_title="FPL Draft Rewards Dashboard", page_icon="⚽", layout="wide")

LEAGUE_DETAILS_URL = "https://draft.premierleague.com/api/league/{}/details"
ENTRY_HISTORY_URL = "https://draft.premierleague.com/api/entry/{}/history"
LIVE_TRACKER_URL = "https://www.anewpla.net/fpl/live/"

# Network settings: a timeout on every request so the app can never hang forever
HEADERS = {"User-Agent": "Mozilla/5.0 (FPL Draft Dashboard)"}
TIMEOUT = 10  # seconds per request

# Prize distribution structure
WEEKLY_PRIZE_MAP = {1: 200, 2: 150, 3: 100, 4: 50}
MOTM_PRIZE = 200
SEASON_1ST_PRIZE = 1000
SEASON_2ND_PRIZE = 500

IST = ZoneInfo("Asia/Kolkata")

# TIEBREAKER RULE:
# If two or more managers are tied on points for a Gameweek (GW) award, or tied
# on total points for the Monthly Manager (MOTM) award, the tie is broken by
# whoever has the higher season-to-date cumulative points total (i.e. their
# running Total through the GW in question / through the last GW of the
# month). There is no prize split: the single manager with the higher
# cumulative total takes the full prize. In the rare case both points and
# cumulative total are still tied, a single winner is chosen deterministically
# (alphabetically) so the prize is never divided.

GW_MONTH_MAPPING = {
    "August": list(range(1, 3)), "September": list(range(3, 6)),
    "October": list(range(6, 10)), "November": list(range(10, 13)),
    "December": list(range(13, 19)), "January": list(range(19, 24)),
    "February": list(range(24, 28)), "March": list(range(28, 31)),
    "April": list(range(31, 34)), "May": list(range(34, 39))
}


def get_current_motm_month():
    """Returns the calendar month name (matching GW_MONTH_MAPPING keys) for
    'today' in IST. Returns None if today's calendar month falls outside the
    season's mapped months (e.g. June/July off-season)."""
    now_ist = datetime.datetime.now(IST)
    month_name = now_ist.strftime("%B")
    return month_name if month_name in GW_MONTH_MAPPING else None


# ==========================================
# DATA FETCHING (PURE API WITH CONCURRENCY, TIMEOUTS & RETRIES)
# ==========================================
def _make_session() -> requests.Session:
    """Session with retry/backoff for transient failures and a browser-like UA."""
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(HEADERS)
    return s


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_league_df(league_id: str) -> pd.DataFrame:
    """Fetches league managers and full Gameweek history concurrently.
    Raises on any failure, so failures are NOT cached by st.cache_data."""
    session = _make_session()
    try:
        res = session.get(LEAGUE_DETAILS_URL.format(league_id), timeout=TIMEOUT)
        res.raise_for_status()
        league_data = res.json()
        entries = league_data.get("league_entries", [])
        if not entries:
            raise ValueError("No teams found in this league.")

        def fetch_entry(entry):
            entry_id = entry["entry_id"]
            manager_name = entry["player_first_name"]
            team_name = entry["entry_name"]

            r = session.get(ENTRY_HISTORY_URL.format(entry_id), timeout=TIMEOUT)
            r.raise_for_status()
            history = r.json().get("history", [])
            return [
                {
                    "entry_id": entry_id,
                    "Teams": manager_name,
                    "Team Name": team_name,
                    "GW": gw["event"],
                    "Points": gw["points"],
                }
                for gw in history
            ]

        all_records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(20, max(1, len(entries)))) as executor:
            for result in executor.map(fetch_entry, entries):
                all_records.extend(result)
    finally:
        session.close()

    if not all_records:
        raise ValueError("No history records retrieved.")
    return pd.DataFrame(all_records)


def load_league_data(league_id: str):
    """Thin wrapper that converts exceptions into (None, error_message)."""
    try:
        return _fetch_league_df(league_id), None
    except requests.exceptions.Timeout:
        return None, "The FPL Draft API timed out. It may be busy or updating; try 'Force Sync' in a minute."
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "unknown"
        return None, f"FPL Draft API returned an error (status {code})."
    except Exception as e:
        return None, f"Could not load FPL data: {e}"


# ==========================================
# TIEBREAK HELPER
# ==========================================
def _cumulative_points_table(raw_df: pd.DataFrame, max_played_gw: int, managers: list) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by manager with columns 1..max_played_gw, where
    each cell is that manager's season-to-date cumulative points total through
    (and including) that Gameweek. Used as the tiebreaker for GW and MOTM awards.
    """
    if not managers or max_played_gw <= 0:
        return pd.DataFrame(index=managers, columns=range(1, max(max_played_gw, 0) + 1)).fillna(0).astype(int)

    # In Draft API, total_points aren't returned per GW in the history endpoint,
    # so we must calculate the running total mathematically using cumsum().
    pivot = (
        raw_df.pivot(index="Teams", columns="GW", values="Points")
        .reindex(index=managers, columns=range(1, max_played_gw + 1))
        .fillna(0)
    )
    return pivot.cumsum(axis=1)


def _resolve_single_winner(score_series: pd.Series, cum_table: pd.DataFrame, tiebreak_gw):
    """
    Given a Series of manager -> points for some period (a GW or a month),
    resolves a single winner:
      1. Highest points wins outright.
      2. Ties are broken by whoever has the higher season-to-date cumulative
         points total through `tiebreak_gw`.
      3. If still tied, a single winner is chosen deterministically
         (alphabetically) so a prize is never split.
    Returns (winner, winning_points, tie_was_broken). winner is None if the
    series is empty.
    """
    if score_series is None or score_series.empty:
        return None, None, False

    top_pts = score_series.max()
    candidates = score_series[score_series == top_pts].index.tolist()
    tie_broken = False

    if len(candidates) > 1 and tiebreak_gw is not None and tiebreak_gw in cum_table.columns:
        cum_scores = {
            w: (cum_table.loc[w, tiebreak_gw] if w in cum_table.index else 0)
            for w in candidates
        }
        top_cum = max(cum_scores.values())
        narrowed = [w for w, v in cum_scores.items() if v == top_cum]
        if len(narrowed) < len(candidates):
            candidates = narrowed
            tie_broken = True

    if len(candidates) > 1:
        candidates = [sorted(candidates)[0]]
        tie_broken = True

    return candidates[0], top_pts, tie_broken


# ==========================================
# DERIVED TABLES (CACHED so widget interactions like the MOTM month
# selector don't force a full recompute on every Streamlit rerun)
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def build_dashboard_tables(raw_df: pd.DataFrame, max_played_gw: int):
    all_gw_cols = [f"GW{i}" for i in range(1, 39)]
    all_managers = sorted(raw_df["Teams"].unique()) if not raw_df.empty else []
    played_gw_cols = [f"GW{i}" for i in range(1, max_played_gw + 1)]

    # Season-to-date cumulative totals per manager, indexed by GW — this is the
    # tiebreaker source for both weekly (GW) awards and the Monthly Manager award.
    cum_table = _cumulative_points_table(raw_df, max_played_gw, all_managers)

    # 1. POINTS MATRIX — nullable Int64 so unplayed GWs stay as <NA> instead of
    # forcing the column to float (which is what produced "None"/decimals before)
    if not raw_df.empty:
        points_pivot = raw_df.pivot(index="Teams", columns="GW", values="Points").reindex(columns=range(1, 39))
    else:
        points_pivot = pd.DataFrame(index=all_managers, columns=range(1, 39))

    points_pivot.columns = all_gw_cols
    points_pivot = points_pivot.astype("Int64")

    if played_gw_cols:
        totals = points_pivot[played_gw_cols].sum(axis=1, skipna=True).astype("Int64")
        play_counts = points_pivot[played_gw_cols].notna().sum(axis=1).replace(0, pd.NA)
        averages = (totals / play_counts).round(0).fillna(0).astype("Int64")
        points_pivot["Total"] = totals
        points_pivot["Average"] = averages
    else:
        points_pivot["Total"] = pd.array([0] * len(points_pivot), dtype="Int64")
        points_pivot["Average"] = pd.array([0] * len(points_pivot), dtype="Int64")

    points_pivot = points_pivot[["Total", "Average"] + all_gw_cols]

    # 2. WEEKLY PODIUM WINNERS
    # Ties on Points for a Gameweek are broken by season-to-date cumulative
    # total through that Gameweek (higher cumulative total wins the tie).
    # If still tied, it sorts Alphabetically so a single manager always wins.
    winners_dict = {pos: {f"GW{gw}": "" for gw in range(1, 39)} for pos in range(1, 5)}
    played_df = raw_df[raw_df["GW"] <= max_played_gw] if max_played_gw else raw_df.iloc[0:0]

    if not played_df.empty:
        for gw in sorted(played_df["GW"].unique()):
            gw_df = played_df[played_df["GW"] == gw].copy()
            if gw in cum_table.columns:
                gw_df["CumTotal"] = gw_df["Teams"].map(cum_table[gw]).fillna(0)
            else:
                gw_df["CumTotal"] = 0

            # Sort by: 1) GW Points desc, 2) CumTotal points desc (Tiebreak), 3) Teams asc (Alphabetical Tiebreak)
            gw_df = gw_df.sort_values(by=["Points", "CumTotal", "Teams"], ascending=[False, False, True])
            top4 = gw_df.head(4)
            for pos, team in enumerate(top4["Teams"].tolist(), start=1):
                winners_dict[pos][f"GW{int(gw)}"] = team

    winners_df = pd.DataFrame(winners_dict).T
    winners_df.index.name = "Winners"

    # 3. MOTM CASH CALCULATION (rounded to whole rupees, no decimals)
    motm_wins_count = {m: 0 for m in all_managers}
    motm_cash_won = {m: 0 for m in all_managers}
    all_month_names = list(GW_MONTH_MAPPING.keys())

    monthly_cash_matrix = pd.DataFrame(pd.NA, index=all_managers, columns=all_month_names, dtype="Int64")
    for month, gws in GW_MONTH_MAPPING.items():
        if all(gw <= max_played_gw for gw in gws):
            monthly_cash_matrix[month] = 0
            m_df = raw_df[raw_df["GW"].isin(gws)]
            if not m_df.empty:
                m_totals = m_df.groupby("Teams")["Points"].sum()
                winner, _, _ = _resolve_single_winner(m_totals, cum_table, max(gws))
                if winner is not None:
                    motm_wins_count[winner] += 1
                    motm_cash_won[winner] += MOTM_PRIZE
                    monthly_cash_matrix.loc[winner, month] = MOTM_PRIZE

    # 4. FINAL STANDINGS CASH
    season_cash_won = {m: 0 for m in all_managers}
    if max_played_gw == 38:
        final_ranks = points_pivot["Total"].sort_values(ascending=False).index.tolist()
        if len(final_ranks) >= 1:
            season_cash_won[final_ranks[0]] += SEASON_1ST_PRIZE
        if len(final_ranks) >= 2:
            season_cash_won[final_ranks[1]] += SEASON_2ND_PRIZE

    # 5. SUMMARY & CASH MATRIX
    # Walk the (small) winners table once instead of assigning cell-by-cell for
    # every manager x gameweek x podium position, which is much slower in pandas.
    counts = {m: {1: 0, 2: 0, 3: 0, 4: 0} for m in all_managers}
    cash_matrix = pd.DataFrame(pd.NA, index=all_managers, columns=all_gw_cols, dtype="Int64")
    if played_gw_cols:
        cash_matrix[played_gw_cols] = 0  # Played Gameweeks default to 0

    for pos in (1, 2, 3, 4):
        for col_name in played_gw_cols:
            winner = winners_dict[pos].get(col_name)
            if winner and winner in counts:
                counts[winner][pos] += 1
                cash_matrix.at[winner, col_name] = WEEKLY_PRIZE_MAP[pos]

    summary_data = []
    for manager in all_managers:
        weekly_amount = sum(counts[manager][p] * WEEKLY_PRIZE_MAP[p] for p in WEEKLY_PRIZE_MAP)
        motm_amount = motm_cash_won[manager]
        season_amount = season_cash_won[manager]
        total_cash = weekly_amount + motm_amount + season_amount

        summary_data.append({
            "Teams": manager,
            "1st": counts[manager][1],
            "2nd": counts[manager][2],
            "3rd": counts[manager][3],
            "4th": counts[manager][4],
            "MOTM Wins": motm_wins_count[manager],
            "Weekly Cash (₹)": int(weekly_amount),
            "MOTM Cash (₹)": int(motm_amount),
            "Season End Cash (₹)": int(season_amount),
            "Total Cash (₹)": int(total_cash)
        })

    if summary_data:
        summary_df = pd.DataFrame(summary_data).set_index("Teams")
        cash_matrix["Total Weekly Cash"] = cash_matrix[played_gw_cols].sum(axis=1) if played_gw_cols else 0
    else:
        summary_df = pd.DataFrame()

    completed_months = [m for m in all_month_names if monthly_cash_matrix[m].notna().any()]
    monthly_cash_matrix["Total MOTM Cash"] = (
        monthly_cash_matrix[completed_months].sum(axis=1, skipna=True).astype("Int64") if completed_months else pd.array([0] * len(all_managers), dtype="Int64")
    )

    return points_pivot, winners_df, summary_df, cash_matrix, monthly_cash_matrix, all_managers, played_gw_cols, cum_table


def _blank_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Render nullable/NaN cells as blank strings without turning played values into decimals."""
    return df.astype(object).where(df.notna(), "")


def highlight_podium(df, winners):
    """Applies CSS background colors to 1st, 2nd, 3rd, and 4th place cells."""
    # Create an empty DataFrame of styles with the same shape/index as the target
    styles = pd.DataFrame('', index=df.index, columns=df.columns)

    # Define exact hex colors requested (with text color for readability)
    colors = {
        1: "background-color: #FFD700; color: black;",  # Gold
        2: "background-color: #C0C0C0; color: black;",  # Silver
        3: "background-color: #CD7F32; color: black;",  # Bronze
        4: "background-color: #4472C4; color: white;"   # Blue
    }

    for col in df.columns:
        if col.startswith("GW") and col in winners.columns:
            for pos in [1, 2, 3, 4]:
                if pos in winners.index:
                    manager = winners.loc[pos, col]
                    if manager and manager in styles.index:
                        styles.loc[manager, col] = colors[pos]

    return styles


# ==========================================
# MONTE CARLO PROJECTION FUNCTIONS (FULLY VECTORIZED)
# ==========================================
def _manager_stats(raw_df, managers):
    """Returns per-manager (mean, std) point stats as numpy arrays.
    Uses a single groupby instead of a per-manager loop."""
    grp = raw_df.groupby("Teams")["Points"]
    mean = grp.mean().reindex(managers).to_numpy(dtype=float)
    std = grp.std(ddof=0).reindex(managers).to_numpy(dtype=float)
    cnt = grp.count().reindex(managers).fillna(0).to_numpy()

    # >=2 games: observed mean/std (std floored at 6); 1 game: that score, std 12; none: 45, std 12
    means = np.where(cnt >= 1, mean, 45.0)
    stds = np.where(cnt >= 2, np.maximum(std, 6.0), 12.0)
    return means, stds


@st.cache_data(ttl=3600, show_spinner=False)
def run_monte_carlo_season_projections(raw_df, managers, max_played_gw, num_simulations=5000):
    managers = list(managers)
    if not managers:
        return {}

    means, stds = _manager_stats(raw_df, managers)
    current_totals = (
        raw_df.groupby("Teams")["Points"].sum().reindex(managers).fillna(0).to_numpy(dtype=float)
    )

    remaining_gws = 38 - max_played_gw
    if remaining_gws <= 0:
        order = np.argsort(-current_totals)
        result = {m: (0, 0) for m in managers}
        top_mgr = managers[order[0]]
        result[top_mgr] = (100, result[top_mgr][1])
        if len(order) > 1:
            second_mgr = managers[order[1]]
            result[second_mgr] = (result[second_mgr][0], 100)
        return result

    # The sum of n independent Normal(mu, sigma) draws is exactly Normal(n*mu, sigma*sqrt(n)),
    # so draw the remaining-season total directly: shape (num_simulations, num_managers).
    rng = np.random.default_rng()
    season_sums = rng.normal(
        loc=means * remaining_gws,
        scale=stds * np.sqrt(remaining_gws),
        size=(num_simulations, len(managers)),
    )
    sim_totals = current_totals[None, :] + np.maximum(season_sums, 0)

    order = np.argsort(-sim_totals, axis=1)
    first_counts = np.bincount(order[:, 0], minlength=len(managers))
    second_counts = (
        np.bincount(order[:, 1], minlength=len(managers)) if len(managers) > 1
        else np.zeros(len(managers), dtype=int)
    )

    return {
        managers[i]: (
            int(round(first_counts[i] / num_simulations * 100)),
            int(round(second_counts[i] / num_simulations * 100))
        )
        for i in range(len(managers))
    }


@st.cache_data(ttl=3600, show_spinner=False)
def run_monte_carlo_motm_projections(raw_df, managers, month_gws, max_played_gw, num_simulations=5000):
    managers = list(managers)
    month_gws = list(month_gws)
    if not managers:
        return {}

    completed_in_month = [gw for gw in month_gws if gw <= max_played_gw]
    remaining_in_month = [gw for gw in month_gws if gw > max_played_gw]

    means, stds = _manager_stats(raw_df, managers)
    current_month_pts = (
        raw_df[raw_df["GW"].isin(completed_in_month)]
        .groupby("Teams")["Points"].sum()
        .reindex(managers).fillna(0).to_numpy(dtype=float)
    )

    if not remaining_in_month:
        top_score = current_month_pts.max()
        winners = current_month_pts == top_score
        win_share = 100 / winners.sum()
        return {managers[i]: (int(round(win_share)) if winners[i] else 0) for i in range(len(managers))}

    rem_count = len(remaining_in_month)
    rng = np.random.default_rng()
    month_sums = rng.normal(
        loc=means * rem_count,
        scale=stds * np.sqrt(rem_count),
        size=(num_simulations, len(managers)),
    )
    sim_totals = current_month_pts[None, :] + np.maximum(month_sums, 0)

    top_scores = sim_totals.max(axis=1, keepdims=True)
    win_mask = sim_totals == top_scores
    win_shares = win_mask / win_mask.sum(axis=1, keepdims=True)
    win_counts = win_shares.sum(axis=0)

    return {
        managers[i]: int(round(win_counts[i] / num_simulations * 100))
        for i in range(len(managers))
    }


# ==========================================
# UI SETUP & DASHBOARD RENDERING
# ==========================================
st.title("⚽ FPL Draft Rewards & Probability Dashboard")

league_id = st.sidebar.text_input("FPL Draft League ID", value="23942", key="league_id_input")

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sync**")
if st.sidebar.button("🔄 Force Sync Latest Points", key="force_sync_button"):
    st.cache_data.clear()
    st.rerun()

if league_id:
    with st.spinner("Fetching live data from FPL Draft API..."):
        raw_df, error = load_league_data(league_id)

    if error:
        st.error(error)
        st.caption("Failed fetches are not cached, so refreshing the page or pressing 'Force Sync' will retry.")
    elif raw_df is not None and not raw_df.empty:

        # Only consider GWs where at least someone actually scored
        gw_sums = raw_df.groupby("GW")["Points"].sum()
        valid_gws = gw_sums[gw_sums > 0].index.tolist()
        raw_df = raw_df[raw_df["GW"].isin(valid_gws)]

        max_played_gw = int(raw_df["GW"].max()) if not raw_df.empty else 0

        points_pivot, winners_df, summary_df, cash_matrix, monthly_cash_matrix, all_managers, played_gw_cols, cum_table = build_dashboard_tables(
            raw_df, max_played_gw
        )

        current_motm_month = get_current_motm_month()
        motm_months_list = list(GW_MONTH_MAPPING.keys())
        default_motm_month = current_motm_month if current_motm_month in motm_months_list else motm_months_list[0]
        default_motm_index = motm_months_list.index(default_motm_month)

        # The month selector lives in the sidebar so it is rendered on every run and
        # its value is available to both the MOTM standings and the probabilities view.
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Manager of the Month**")
        selected_month = st.sidebar.selectbox(
            "Select Calendar Month", motm_months_list, index=default_motm_index, key="motm_month_selectbox"
        )
        target_gws = GW_MONTH_MAPPING[selected_month]

        # ==========================================
        # LIVE TICKER — current GW top 4 + current month leader
        # ==========================================
        current_gw_top4 = []  # list of (position, manager, points)
        if max_played_gw and f"GW{max_played_gw}" in winners_df.columns:
            gw_col = f"GW{max_played_gw}"
            for pos in [1, 2, 3, 4]:
                manager = winners_df.loc[pos, gw_col]
                if manager:
                    pts = int(points_pivot.loc[manager, gw_col]) if manager in points_pivot.index else None
                    current_gw_top4.append((pos, manager, pts))

        current_month_name, current_month_leader, current_month_points, current_month_complete = None, None, None, False
        if max_played_gw:
            for month, gws in GW_MONTH_MAPPING.items():
                if max_played_gw in gws:
                    current_month_name = month
                    current_month_complete = all(gw <= max_played_gw for gw in gws)
                    played_gws_this_month = [gw for gw in gws if gw <= max_played_gw]
                    m_df_current = raw_df[raw_df["GW"].isin(played_gws_this_month)]
                    if not m_df_current.empty:
                        m_totals_current = m_df_current.groupby("Teams")["Points"].sum()
                        tiebreak_gw_current = max(played_gws_this_month)
                        winner, pts, _ = _resolve_single_winner(m_totals_current, cum_table, tiebreak_gw_current)
                        current_month_leader = winner
                        current_month_points = int(pts) if pts is not None else None
                    break

        POSITION_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉", 4: "🎖️"}

        ticker_slides = []

        if current_gw_top4:
            gw_parts = "&nbsp;&nbsp;&nbsp;".join(
                f"{POSITION_MEDALS[pos]} <strong>{manager}</strong> ({pts} pts)"
                for pos, manager, pts in current_gw_top4
            )
            ticker_slides.append(f"🔥 GW{max_played_gw} Top 4:&nbsp;&nbsp;&nbsp;{gw_parts}")

        if current_month_leader:
            month_label = "Winner" if current_month_complete else "Leader"
            ticker_slides.append(
                f"👑 {current_month_name} MOTM {month_label}: <strong>{current_month_leader}</strong> ({current_month_points} pts)"
            )

        if not summary_df.empty and "Total Cash (₹)" in summary_df.columns:
            cash_sorted = summary_df.sort_values(by="Total Cash (₹)", ascending=False)
            cash_parts = "&nbsp;&nbsp;&nbsp;".join(
                f"<strong>{manager}</strong> (₹{int(row['Total Cash (₹)'])})"
                for manager, row in cash_sorted.iterrows()
            )
            ticker_slides.append(f"💰 Total Cash Won:&nbsp;&nbsp;&nbsp;{cash_parts}")

        if ticker_slides:
            slides_json = json.dumps(ticker_slides)
            st.iframe(
                f"""
                <div style="overflow: hidden; white-space: nowrap; background: linear-gradient(90deg,#37003c,#00ff85);
                            border-radius: 8px; padding: 12px 0; box-sizing: border-box; margin: 0;
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                    <div id="ticker-slide" style="display: inline-block; white-space: nowrap; padding-left: 100%;
                                color: #ffffff; font-size: 16px; font-weight: 600;"></div>
                </div>
                <style>
                @keyframes ticker-scroll {{
                    0%   {{ transform: translateX(0); }}
                    100% {{ transform: translateX(-100%); }}
                }}
                </style>
                <script>
                const tickerSlides = {slides_json};
                const scrollSeconds = 35; // slower, more readable scroll speed
                let tickerIdx = 0;
                const tickerEl = document.getElementById('ticker-slide');

                function showTickerSlide(i) {{
                    tickerEl.style.animation = 'none';
                    tickerEl.innerHTML = tickerSlides[i];
                    void tickerEl.offsetWidth; // force reflow so the animation restarts cleanly
                    tickerEl.style.animation = `ticker-scroll ${{scrollSeconds}}s linear infinite`;
                }}

                showTickerSlide(tickerIdx);
                setInterval(() => {{
                    tickerIdx = (tickerIdx + 1) % tickerSlides.length;
                    showTickerSlide(tickerIdx);
                }}, scrollSeconds * 1000);
                </script>
                """,
                height=54,
            )

        # ==========================================
        # DASHBOARD VIEWS
        # st.tabs runs ALL tabs on every rerun (including the Monte Carlo
        # simulations and the external live-tracker iframe). A radio selector only
        # renders the chosen view, so each run does far less work.
        # ==========================================
        VIEW_OVERVIEW = "📊 Points & Standings"
        VIEW_CASH = "💰 Podium Counts & Cash Won"
        VIEW_PROB = "🎲 Win Probabilities (%)"
        VIEW_LIVE = "🔴 Live GW Points"

        view = st.radio(
            "View",
            [VIEW_OVERVIEW, VIEW_CASH, VIEW_PROB, VIEW_LIVE],
            horizontal=True,
            label_visibility="collapsed",
            key="dashboard_view",
        )

        if view == VIEW_OVERVIEW:
            st.subheader("📋 Points Matrix (GW1 - GW38)")
            st.caption(
                f"Sorted by GW{max_played_gw} points (latest Gameweek), highest to lowest; ties are broken by "
                f"season-to-date total points, then alphabetically. Color codes indicate 1st - Gold, "
                f"2nd - Silver, 3rd - Bronze, and 4th - Blue."
                if max_played_gw else ""
            )

            if max_played_gw and f"GW{max_played_gw}" in points_pivot.columns:
                # Sort by: 1) latest GW points desc, 2) season-to-date cumulative total desc (tiebreak),
                # 3) manager name asc. This mirrors the podium-winner logic so the row order
                # always matches the gold/silver/bronze/blue colouring.
                gw_col_name = f"GW{max_played_gw}"
                sort_df = pd.DataFrame({
                    "gw": points_pivot[gw_col_name].astype(float),
                    "cum": cum_table[max_played_gw].reindex(points_pivot.index).fillna(0),
                    "name": points_pivot.index,
                })
                sorted_order = sort_df.sort_values(
                    by=["gw", "cum", "name"],
                    ascending=[False, False, True],
                    na_position="last",
                ).index
                points_pivot_display = points_pivot.loc[sorted_order]
            else:
                points_pivot_display = points_pivot

            # Clean nulls first, then apply the highlighting style
            display_df = _blank_nulls(points_pivot_display)
            styled_points = display_df.style.apply(
                highlight_podium,
                winners=winners_df,
                axis=None
            )

            st.dataframe(styled_points, width="stretch")

            st.subheader("🏆 Weekly Podium Winners (1st - 4th)")
            st.caption("Ties on GW points are broken by season-to-date cumulative total points.")
            st.dataframe(winners_df, width="stretch")

            st.markdown("---")
            st.subheader("👑 Manager of the Month Standings")
            if selected_month == current_motm_month:
                st.caption(f"{selected_month} is the current month (IST). Change the month from the sidebar.")
            else:
                st.caption(f"Showing {selected_month}. Change the month from the sidebar.")

            is_month_complete = all(gw <= max_played_gw for gw in target_gws)
            if is_month_complete:
                st.write("✅ **Month Completed (Prize Awarded)**")
            else:
                st.write(f"⏳ **In Progress / Upcoming** (Gameweeks: {target_gws})")

            motm_filtered = raw_df[raw_df["GW"].isin(target_gws)]
            if not motm_filtered.empty:
                motm_pivot = motm_filtered.pivot(index="Teams", columns="GW", values="Points")

                for gw in target_gws:
                    if gw not in motm_pivot.columns:
                        motm_pivot[gw] = np.nan

                motm_pivot = motm_pivot[target_gws].astype("Int64")
                motm_pivot["Total Points"] = motm_pivot.sum(axis=1, skipna=True).astype("Int64")

                motm_pivot = motm_pivot.sort_values(by="Total Points", ascending=False).reset_index()
                motm_pivot.index += 1

                rename_cols = {gw: f"GW{gw}" for gw in target_gws}
                motm_pivot = motm_pivot.rename(columns=rename_cols)

                m_totals_for_month = motm_filtered.groupby("Teams")["Points"].sum()
                played_target_gws = [gw for gw in target_gws if gw <= max_played_gw]
                tiebreak_gw = max(played_target_gws) if played_target_gws else None
                winner_name, top_score, tie_broken_by_total = _resolve_single_winner(
                    m_totals_for_month, cum_table, tiebreak_gw
                )
                tie_note = " (tie broken by season-to-date total points)" if tie_broken_by_total else ""

                if is_month_complete:
                    st.success(
                        f"🎉 Official MOTM Winner: {winner_name} with {int(top_score)} pts "
                        f"(Won ₹{MOTM_PRIZE}){tie_note}!"
                    )
                else:
                    st.info(f"Leader so far: {winner_name} ({int(top_score)} pts){tie_note}. Cash will be awarded after all GWs finish.")

                st.dataframe(_blank_nulls(motm_pivot), width="stretch")
            else:
                st.info(f"No Gameweek points finalized yet for {selected_month} (GWs: {target_gws}).")

        elif view == VIEW_CASH:
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.subheader("🎖️ Podium Counts & Total Cash Won")
                if not summary_df.empty:
                    st.dataframe(summary_df.sort_values(by="Total Cash (₹)", ascending=False), width="stretch")
            with col_right:
                st.subheader("💵 Prize Rules")
                prize_rule_df = pd.DataFrame({
                    "Award Category": ["Weekly 1st", "Weekly 2nd", "Weekly 3rd", "Weekly 4th", "MOTM (Complete)", "1st Overall", "2nd Overall"],
                    "Cash": [f"₹{WEEKLY_PRIZE_MAP[1]}", f"₹{WEEKLY_PRIZE_MAP[2]}", f"₹{WEEKLY_PRIZE_MAP[3]}", f"₹{WEEKLY_PRIZE_MAP[4]}", f"₹{MOTM_PRIZE}", f"₹{SEASON_1ST_PRIZE}", f"₹{SEASON_2ND_PRIZE}"]
                })
                st.dataframe(prize_rule_df, width="stretch", hide_index=True)
                st.caption("Ties (GW or MOTM) are broken by higher season-to-date cumulative points.")

            st.markdown("---")
            st.subheader("💳 Weekly Cash Won per Gameweek (₹)")
            st.dataframe(_blank_nulls(cash_matrix), width="stretch")

            st.markdown("---")
            st.subheader("📅 Monthly Cash Won per Month (₹)")
            st.caption("Blank = month not yet complete, so MOTM cash hasn't been awarded. Ties are broken by higher season-to-date cumulative points.")
            st.dataframe(_blank_nulls(monthly_cash_matrix), width="stretch")

        elif view == VIEW_PROB:
            st.header("🎲 Monte Carlo Win Probability Projections")
            st.caption("Projections based on 5,000 simulations using each manager's historical scoring rate and variance.")
            if all_managers:
                col_seas, col_month = st.columns(2)
                with col_seas:
                    st.subheader("🏆 End-of-Season Probability (GW38)")
                    season_probs = run_monte_carlo_season_projections(raw_df, tuple(all_managers), max_played_gw)
                    season_prob_df = pd.DataFrame([{"Teams": m, "Current Pts": int(points_pivot.loc[m, "Total"]) if "Total" in points_pivot.columns else 0, "1st Place (%)": f"{season_probs.get(m, (0, 0))[0]}%", "2nd Place (%)": f"{season_probs.get(m, (0, 0))[1]}%"} for m in all_managers]).sort_values(by="Current Pts", ascending=False).reset_index(drop=True)
                    st.dataframe(season_prob_df, width="stretch", hide_index=True)
                with col_month:
                    st.subheader(f"👑 MOTM Probability: {selected_month}")
                    motm_probs = run_monte_carlo_motm_projections(raw_df, tuple(all_managers), tuple(target_gws), max_played_gw)
                    motm_prob_df = pd.DataFrame([{"Teams": m, "Win MOTM Prob (%)": f"{motm_probs.get(m, 0)}%"} for m in all_managers]).sort_values(by="Win MOTM Prob (%)", ascending=False).reset_index(drop=True)
                    st.dataframe(motm_prob_df, width="stretch", hide_index=True)

        elif view == VIEW_LIVE:
            st.subheader("🔴 Live Gameweek Points Tracker")
            st.markdown(f"Monitor live points. *(If the embed fails, [click here to open the tracker]({LIVE_TRACKER_URL}))*")
            # Only created when this view is selected, so the external page isn't loaded on every run.
            st.iframe(LIVE_TRACKER_URL, height=800)
else:
    st.info("👈 Please enter your Draft League ID in the sidebar to load the dashboard.")
