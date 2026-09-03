import requests
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import concurrent.futures

# --- Page Configuration ---
st.set_page_config(page_title="FPL Draft Rewards Dashboard", page_icon="⚽", layout="wide")

LEAGUE_DETAILS_URL = "https://draft.premierleague.com/api/league/{}/details"
ENTRY_HISTORY_URL = "https://draft.premierleague.com/api/entry/{}/history"

# Prize distribution structure
WEEKLY_PRIZE_MAP = {1: 200, 2: 150, 3: 100, 4: 50}
MOTM_PRIZE = 200
SEASON_1ST_PRIZE = 1000
SEASON_2ND_PRIZE = 500

# TIEBREAKER RULE:
# If two or more managers are tied on points for a Gameweek (GW) award, or tied
# on total points for the Monthly Manager (MOTM) award, the tie is broken by
# whoever has the higher season-to-date cumulative points total (i.e. their
# running Total through the GW in question / through the last GW of the
# month). If managers are still tied after applying this tiebreaker, the
# prize is split equally among the managers still tied.

GW_MONTH_MAPPING = {
    "August": list(range(1, 3)), "September": list(range(3, 6)),
    "October": list(range(6, 10)), "November": list(range(10, 13)),
    "December": list(range(13, 19)), "January": list(range(19, 24)),
    "February": list(range(24, 28)), "March": list(range(28, 31)),
    "April": list(range(31, 34)), "May": list(range(34, 39))
}

# ==========================================
# DATA FETCHING (PURE API WITH CONCURRENCY)
# ==========================================
@st.cache_data(ttl=3600)
def load_league_data(league_id: str):
    """Fetches league managers and full Gameweek history concurrently from FPL API."""
    try:
        with requests.Session() as session:
            res = session.get(LEAGUE_DETAILS_URL.format(league_id))
            if res.status_code != 200:
                return None, f"Failed to retrieve league details (Status code: {res.status_code})"

            league_data = res.json()
            entries = league_data.get("league_entries", [])
            if not entries:
                return None, "No teams found in this league."

            def fetch_entry(entry):
                entry_id = entry["entry_id"]
                manager_name = entry["player_first_name"]
                team_name = entry["entry_name"]

                records = []
                hist_res = session.get(ENTRY_HISTORY_URL.format(entry_id))
                if hist_res.status_code == 200:
                    history = hist_res.json().get("history", [])
                    for gw in history:
                        records.append({
                            "entry_id": entry_id,
                            "Teams": manager_name,
                            "Team Name": team_name,
                            "GW": gw["event"],
                            "Points": gw["points"]
                        })
                return records

            all_records = []
            # Reused session gives connection-pooling/keep-alive benefits across threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(20, max(1, len(entries)))) as executor:
                for result in executor.map(fetch_entry, entries):
                    all_records.extend(result)

        if not all_records:
            return None, "No history records retrieved."

        df = pd.DataFrame(all_records)
        return df, None
    except Exception as e:
        return None, str(e)


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

    pivot = (
        raw_df.pivot(index="Teams", columns="GW", values="Points")
        .reindex(index=managers, columns=range(1, max_played_gw + 1))
        .fillna(0)
    )
    return pivot.cumsum(axis=1)


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
    winners_dict = {pos: {f"GW{gw}": "" for gw in range(1, 39)} for pos in range(1, 5)}
    played_df = raw_df[raw_df["GW"] <= max_played_gw] if max_played_gw else raw_df.iloc[0:0]
    if not played_df.empty:
        for gw in sorted(played_df["GW"].unique()):
            gw_df = played_df[played_df["GW"] == gw].copy()
            if gw in cum_table.columns:
                gw_df["CumTotal"] = gw_df["Teams"].map(cum_table[gw]).fillna(0)
            else:
                gw_df["CumTotal"] = 0
            # Sort by GW Points desc, then season-to-date cumulative total desc (tiebreak)
            gw_df = gw_df.sort_values(by=["Points", "CumTotal"], ascending=[False, False])
            top4 = gw_df.head(4)
            for pos, team in enumerate(top4["Teams"].tolist(), start=1):
                winners_dict[pos][f"GW{int(gw)}"] = team

    winners_df = pd.DataFrame(winners_dict).T
    winners_df.index.name = "Winners"

    # 3. MOTM CASH CALCULATION (rounded to whole rupees, no decimals)
    # Ties on total Points for the month are broken by season-to-date cumulative
    # total through the last GW of that month. If still tied after that, the
    # MOTM prize is split equally among those still tied.
    motm_wins_count = {m: 0 for m in all_managers}
    motm_cash_won = {m: 0 for m in all_managers}
    for month, gws in GW_MONTH_MAPPING.items():
        if all(gw <= max_played_gw for gw in gws):
            m_df = raw_df[raw_df["GW"].isin(gws)]
            if not m_df.empty:
                m_totals = m_df.groupby("Teams")["Points"].sum()
                top_pts = m_totals.max()
                month_winners = m_totals[m_totals == top_pts].index.tolist()

                if len(month_winners) > 1:
                    last_gw = max(gws)
                    if last_gw in cum_table.columns:
                        cum_scores = {
                            w: (cum_table.loc[w, last_gw] if w in cum_table.index else 0)
                            for w in month_winners
                        }
                        top_cum = max(cum_scores.values())
                        month_winners = [w for w, v in cum_scores.items() if v == top_cum]

                prize_per_mgr = round(MOTM_PRIZE / len(month_winners))
                for w in month_winners:
                    motm_wins_count[w] += 1
                    motm_cash_won[w] += prize_per_mgr

    # 4. FINAL STANDINGS CASH
    season_cash_won = {m: 0 for m in all_managers}
    if max_played_gw == 38:
        final_ranks = points_pivot["Total"].sort_values(ascending=False).index.tolist()
        if len(final_ranks) >= 1:
            season_cash_won[final_ranks[0]] += SEASON_1ST_PRIZE
        if len(final_ranks) >= 2:
            season_cash_won[final_ranks[1]] += SEASON_2ND_PRIZE

    # 5. SUMMARY & CASH MATRIX — nullable Int64 so unplayed GWs stay blank, not "None"
    summary_data = []
    cash_matrix = pd.DataFrame(pd.NA, index=all_managers, columns=all_gw_cols, dtype="Int64")

    for manager in all_managers:
        counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for gw in range(1, max_played_gw + 1):
            col_name = f"GW{gw}"
            cash_matrix.loc[manager, col_name] = 0  # Default played Gameweek to 0
            for pos in [1, 2, 3, 4]:
                if winners_dict[pos].get(col_name) == manager:
                    counts[pos] += 1
                    cash_matrix.loc[manager, col_name] = WEEKLY_PRIZE_MAP[pos]

        weekly_amount = sum(counts[p] * WEEKLY_PRIZE_MAP[p] for p in WEEKLY_PRIZE_MAP)
        motm_amount = motm_cash_won[manager]
        season_amount = season_cash_won[manager]
        total_cash = weekly_amount + motm_amount + season_amount

        summary_data.append({
            "Teams": manager,
            "1st (GW)": counts[1],
            "2nd (GW)": counts[2],
            "3rd (GW)": counts[3],
            "4th (GW)": counts[4],
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

    return points_pivot, winners_df, summary_df, cash_matrix, all_managers, played_gw_cols, cum_table


def _blank_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Render nullable/NaN cells as blank strings without turning played values into decimals."""
    return df.astype(object).where(df.notna(), "")


# ==========================================
# MONTE CARLO PROJECTION FUNCTIONS (VECTORIZED)
# ==========================================
def _manager_stats(raw_df, managers):
    """Returns per-manager (mean, std) point stats as numpy arrays."""
    means = np.empty(len(managers))
    stds = np.empty(len(managers))
    for i, mgr in enumerate(managers):
        pts = raw_df.loc[raw_df["Teams"] == mgr, "Points"].values
        if len(pts) >= 2:
            means[i], stds[i] = pts.mean(), max(pts.std(), 6.0)
        elif len(pts) == 1:
            means[i], stds[i] = pts[0], 12.0
        else:
            means[i], stds[i] = 45.0, 12.0
    return means, stds


@st.cache_data(ttl=3600, show_spinner=False)
def run_monte_carlo_season_projections(raw_df, managers, max_played_gw, num_simulations=5000):
    managers = list(managers)
    if not managers:
        return {}

    means, stds = _manager_stats(raw_df, managers)
    current_totals = np.array([
        raw_df.loc[raw_df["Teams"] == mgr, "Points"].values.sum() for mgr in managers
    ], dtype=float)

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

    # Simulate all managers & all simulations at once: (num_simulations, num_managers, remaining_gws)
    sims = np.random.normal(
        loc=means[None, :, None],
        scale=stds[None, :, None],
        size=(num_simulations, len(managers), remaining_gws)
    )
    sim_totals = current_totals[None, :] + np.maximum(sims.sum(axis=2), 0)

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
    current_month_pts = np.array([
        raw_df.loc[(raw_df["Teams"] == mgr) & (raw_df["GW"].isin(completed_in_month)), "Points"].values.sum()
        for mgr in managers
    ], dtype=float)

    if not remaining_in_month:
        top_score = current_month_pts.max()
        winners = current_month_pts == top_score
        win_share = 100 / winners.sum()
        return {managers[i]: (int(round(win_share)) if winners[i] else 0) for i in range(len(managers))}

    rem_count = len(remaining_in_month)
    sims = np.random.normal(
        loc=means[None, :, None],
        scale=stds[None, :, None],
        size=(num_simulations, len(managers), rem_count)
    )
    sim_totals = current_month_pts[None, :] + np.maximum(sims.sum(axis=2), 0)

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

league_id = st.sidebar.text_input("FPL Draft League ID", value="23942")

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sync**")
if st.sidebar.button("🔄 Force Sync Latest Points"):
    st.cache_data.clear()
    st.rerun()

if league_id:
    with st.spinner("Fetching live data from FPL Draft API..."):
        raw_df, error = load_league_data(league_id)

    if error:
        st.error(error)
    elif raw_df is not None and not raw_df.empty:

        # Only consider GWs where at least someone actually scored
        gw_sums = raw_df.groupby("GW")["Points"].sum()
        valid_gws = gw_sums[gw_sums > 0].index.tolist()
        raw_df = raw_df[raw_df["GW"].isin(valid_gws)]

        max_played_gw = int(raw_df["GW"].max()) if not raw_df.empty else 0

        points_pivot, winners_df, summary_df, cash_matrix, all_managers, played_gw_cols, cum_table = build_dashboard_tables(
            raw_df, max_played_gw
        )

        # ==========================================
        # DASHBOARD TABS
        # ==========================================
        tab_overview, tab_cash, tab_motm, tab_prob, tab_live = st.tabs([
            "📊 Points & Standings", "💰 Podium Counts & Cash Won", "👑 Manager of the Month", "🎲 Win Probabilities (%)", "🔴 Live GW Points"
        ])

        with tab_overview:
            st.subheader("📋 Points Matrix (GW1 - GW38)")
            st.dataframe(_blank_nulls(points_pivot), use_container_width=True)

            st.subheader("🏆 Weekly Podium Winners (1st - 4th)")
            st.caption("Ties on GW points are broken by season-to-date cumulative total points.")
            st.dataframe(winners_df, use_container_width=True)

        with tab_cash:
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.subheader("🎖️ Podium Counts & Total Cash Won")
                if not summary_df.empty:
                    st.dataframe(summary_df.sort_values(by="Total Cash (₹)", ascending=False), use_container_width=True)
            with col_right:
                st.subheader("💵 Prize Rules")
                prize_rule_df = pd.DataFrame({
                    "Award Category": ["Weekly 1st", "Weekly 2nd", "Weekly 3rd", "Weekly 4th", "MOTM (Complete)", "1st Overall", "2nd Overall"],
                    "Cash": [f"₹{WEEKLY_PRIZE_MAP[1]}", f"₹{WEEKLY_PRIZE_MAP[2]}", f"₹{WEEKLY_PRIZE_MAP[3]}", f"₹{WEEKLY_PRIZE_MAP[4]}", f"₹{MOTM_PRIZE}", f"₹{SEASON_1ST_PRIZE}", f"₹{SEASON_2ND_PRIZE}"]
                })
                st.dataframe(prize_rule_df, use_container_width=True, hide_index=True)
                st.caption("Ties (GW or MOTM) are broken by higher season-to-date cumulative points.")

            st.markdown("---")
            st.subheader("💳 Weekly Cash Won per Gameweek (₹)")
            st.dataframe(_blank_nulls(cash_matrix), use_container_width=True)

        with tab_motm:
            st.subheader("👑 Manager of the Month Standings")
            selected_month = st.selectbox("Select Calendar Month", list(GW_MONTH_MAPPING.keys()))
            target_gws = GW_MONTH_MAPPING[selected_month]

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

                top_score = motm_pivot.iloc[0]["Total Points"]
                current_leaders = motm_pivot[motm_pivot["Total Points"] == top_score]["Teams"].tolist()
                tie_broken_by_total = False

                # Apply the season-to-date cumulative-total tiebreak whenever there's
                # a tie at the top, using the cumulative total through the latest
                # played GW within this month.
                if len(current_leaders) > 1:
                    played_target_gws = [gw for gw in target_gws if gw <= max_played_gw]
                    tiebreak_gw = max(played_target_gws) if played_target_gws else None
                    if tiebreak_gw is not None and tiebreak_gw in cum_table.columns:
                        cum_scores = {
                            w: (cum_table.loc[w, tiebreak_gw] if w in cum_table.index else 0)
                            for w in current_leaders
                        }
                        top_cum = max(cum_scores.values())
                        tiebroken_leaders = [w for w, v in cum_scores.items() if v == top_cum]
                        if len(tiebroken_leaders) < len(current_leaders):
                            current_leaders = tiebroken_leaders
                            tie_broken_by_total = True

                tie_note = " (tie broken by season-to-date total points)" if tie_broken_by_total else ""

                if is_month_complete:
                    st.success(
                        f"🎉 Official MOTM Winner(s): {', '.join(current_leaders)} with {int(top_score)} pts "
                        f"(Won ₹{round(MOTM_PRIZE / len(current_leaders))} each){tie_note}!"
                    )
                else:
                    st.info(f"Leader so far: {', '.join(current_leaders)} ({int(top_score)} pts){tie_note}. Cash will be awarded after all GWs finish.")

                st.dataframe(_blank_nulls(motm_pivot), use_container_width=True)
            else:
                st.info(f"No Gameweek points finalized yet for {selected_month} (GWs: {target_gws}).")

        with tab_prob:
            st.header("🎲 Monte Carlo Win Probability Projections")
            st.caption("Projections based on 5,000 simulations using each manager's historical scoring rate and variance.")
            if all_managers:
                col_seas, col_month = st.columns(2)
                with col_seas:
                    st.subheader("🏆 End-of-Season Probability (GW38)")
                    season_probs = run_monte_carlo_season_projections(raw_df, tuple(all_managers), max_played_gw)
                    season_prob_df = pd.DataFrame([{"Teams": m, "Current Pts": int(points_pivot.loc[m, "Total"]) if "Total" in points_pivot.columns else 0, "1st Place (%)": f"{season_probs.get(m, (0, 0))[0]}%", "2nd Place (%)": f"{season_probs.get(m, (0, 0))[1]}%"} for m in all_managers]).sort_values(by="Current Pts", ascending=False).reset_index(drop=True)
                    st.dataframe(season_prob_df, use_container_width=True, hide_index=True)
                with col_month:
                    st.subheader(f"👑 MOTM Probability: {selected_month}")
                    motm_probs = run_monte_carlo_motm_projections(raw_df, tuple(all_managers), tuple(target_gws), max_played_gw)
                    motm_prob_df = pd.DataFrame([{"Teams": m, "Win MOTM Prob (%)": f"{motm_probs.get(m, 0)}%"} for m in all_managers]).sort_values(by="Win MOTM Prob (%)", ascending=False).reset_index(drop=True)
                    st.dataframe(motm_prob_df, use_container_width=True, hide_index=True)

        with tab_live:
            st.subheader("🔴 Live Gameweek Points Tracker")
            st.markdown("Monitor live points. *(If the embed fails, [click here to open the tracker](https://www.anewpla.net/fpl/live/))*")

            components.html(
                """
                <iframe 
                    src="https://www.anewpla.net/fpl/live/" 
                    width="100%" 
                    height="800px" 
                    style="border:none;" 
                    loading="lazy" 
                    title="Live FPL Gameweek Points Tracker">
                </iframe>
                """,
                height=800,
                scrolling=True
            )
else:
    st.info("👈 Please enter your Draft League ID in the sidebar to load the dashboard.")
