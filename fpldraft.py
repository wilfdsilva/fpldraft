import os
import sqlite3
import requests
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import concurrent.futures
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="FPL Draft Rewards Dashboard", page_icon="⚽", layout="wide")

LEAGUE_ID = "23942"
DB_NAME = "fpl_draft.db"
LEAGUE_DETAILS_URL = "https://draft.premierleague.com/api/league/{}/details"
ENTRY_HISTORY_URL = "https://draft.premierleague.com/api/entry/{}/history"

# Prize distribution structure
WEEKLY_PRIZE_MAP = {1: 200, 2: 150, 3: 100, 4: 50}
MOTM_PRIZE = 200
SEASON_1ST_PRIZE = 1000
SEASON_2ND_PRIZE = 500

GW_MONTH_MAPPING = {
    "August": list(range(1, 3)), "September": list(range(3, 6)),
    "October": list(range(6, 10)), "November": list(range(10, 13)),
    "December": list(range(13, 19)), "January": list(range(19, 24)),
    "February": list(range(24, 28)), "March": list(range(28, 31)),
    "April": list(range(31, 34)), "May": list(range(34, 39))
}

# ==========================================
# BACKGROUND SYNC & DATABASE FUNCTIONS
# ==========================================
def sync_fpl_draft_db(league_id=LEAGUE_ID):
    """Fetches FPL API data concurrently and updates the local SQLite database."""
    res = requests.get(LEAGUE_DETAILS_URL.format(league_id))
    if res.status_code != 200:
        return False, f"Failed to retrieve league details (Status code: {res.status_code})"

    league_data = res.json()
    entries = league_data.get("league_entries", [])
    if not entries:
        return False, "No teams found in this league."

    def fetch_entry(entry):
        entry_id = entry["entry_id"]
        manager_name = entry["player_first_name"]
        team_name = entry["entry_name"]
        
        records = []
        hist_res = requests.get(ENTRY_HISTORY_URL.format(entry_id))
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
    # Utilize concurrent fetching for faster API performance
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for result in executor.map(fetch_entry, entries):
            all_records.extend(result)

    if not all_records:
        return False, "No history records retrieved."

    df = pd.DataFrame(all_records)

    # Save to SQLite database
    conn = sqlite3.connect(DB_NAME)
    df.to_sql("gw_points", conn, if_exists="replace", index=False)
    
    # Store last synced timestamp
    meta_df = pd.DataFrame([{"last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
    meta_df.to_sql("metadata", conn, if_exists="replace", index=False)
    conn.close()
    
    # Invalidate Streamlit's cache to force it to read the new DB data
    load_data_from_db.clear()
    
    return True, "Database successfully updated."

def check_and_auto_sync():
    """Checks if the DB is missing or older than 1 hour. If so, triggers an automatic sync."""
    if not os.path.exists(DB_NAME):
        sync_fpl_draft_db()
        return
        
    try:
        conn = sqlite3.connect(DB_NAME)
        meta_df = pd.read_sql("SELECT * FROM metadata", conn)
        conn.close()
        last_updated_str = meta_df.iloc[0]["last_updated"]
        last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
        
        # If DB is older than 60 minutes, fetch fresh data automatically
        if datetime.now() - last_updated > timedelta(minutes=60):
            sync_fpl_draft_db()
    except Exception:
        sync_fpl_draft_db()

@st.cache_data(ttl=3600)
def load_data_from_db():
    """Reads league data quickly from the local SQLite database."""
    if not os.path.exists(DB_NAME):
        return None, None, "Database file not found."
    
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql("SELECT * FROM gw_points", conn)
        meta_df = pd.read_sql("SELECT * FROM metadata", conn)
        last_updated_str = meta_df.iloc[0]["last_updated"] if not meta_df.empty else "Unknown"
        conn.close()
        return df, last_updated_str, None
    except Exception as e:
        return None, None, str(e)

# ==========================================
# MONTE CARLO PROJECTION FUNCTIONS
# ==========================================
def run_monte_carlo_season_projections(raw_df, managers, max_played_gw, num_simulations=5000):
    stats = {}
    current_totals = {}
    for mgr in managers:
        mgr_pts = raw_df[raw_df["Teams"] == mgr]["Points"].values
        current_totals[mgr] = float(mgr_pts.sum()) if len(mgr_pts) > 0 else 0.0
        
        if len(mgr_pts) >= 2:
            stats[mgr] = (float(np.mean(mgr_pts)), max(float(np.std(mgr_pts)), 6.0))
        elif len(mgr_pts) == 1:
            stats[mgr] = (float(mgr_pts[0]), 12.0)
        else:
            stats[mgr] = (45.0, 12.0)

    remaining_gws = 38 - max_played_gw
    if remaining_gws <= 0:
        sorted_mgrs = sorted(current_totals.items(), key=lambda x: x[1], reverse=True)
        if not sorted_mgrs: return {}
        return {
            m: (100.0 if m == sorted_mgrs[0][0] else 0.0, 100.0 if len(sorted_mgrs) > 1 and m == sorted_mgrs[1][0] else 0.0)
            for m in managers
        }

    first_counts = {m: 0 for m in managers}
    second_counts = {m: 0 for m in managers}

    for _ in range(num_simulations):
        sim_scores = {}
        for m in managers:
            mean, std = stats[m]
            sim_pts = np.sum(np.random.normal(mean, std, remaining_gws))
            sim_scores[m] = current_totals[m] + max(0, sim_pts)

        ranked = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
        first_counts[ranked[0][0]] += 1
        if len(ranked) > 1:
            second_counts[ranked[1][0]] += 1

    return {
        m: (round((first_counts[m] / num_simulations) * 100, 1), round((second_counts[m] / num_simulations) * 100, 1))
        for m in managers
    }


def run_monte_carlo_motm_projections(raw_df, managers, month_gws, max_played_gw, num_simulations=5000):
    completed_in_month = [gw for gw in month_gws if gw <= max_played_gw]
    remaining_in_month = [gw for gw in month_gws if gw > max_played_gw]
    current_month_pts = {}
    stats = {}

    for mgr in managers:
        pts = raw_df[(raw_df["Teams"] == mgr) & (raw_df["GW"].isin(completed_in_month))]["Points"].values
        current_month_pts[mgr] = float(pts.sum()) if len(pts) > 0 else 0.0
        
        all_pts = raw_df[raw_df["Teams"] == mgr]["Points"].values
        if len(all_pts) >= 2:
            stats[mgr] = (float(np.mean(all_pts)), max(float(np.std(all_pts)), 6.0))
        elif len(all_pts) == 1:
            stats[mgr] = (float(all_pts[0]), 12.0)
        else:
            stats[mgr] = (45.0, 12.0)

    if not remaining_in_month:
        sorted_m = sorted(current_month_pts.items(), key=lambda x: x[1], reverse=True)
        if not sorted_m: return {m: 0.0 for m in managers}
        top_score = sorted_m[0][1]
        winners = [m for m, pts in sorted_m if pts == top_score]
        return {m: (100.0 / len(winners) if m in winners else 0.0) for m in managers}

    win_counts = {m: 0 for m in managers}
    rem_count = len(remaining_in_month)

    for _ in range(num_simulations):
        sim_scores = {}
        for m in managers:
            mean, std = stats[m]
            sim_pts = np.sum(np.random.normal(mean, std, rem_count))
            sim_scores[m] = current_month_pts[m] + max(0, sim_pts)

        ranked = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
        top_score = ranked[0][1]
        sim_winners = [m for m, pts in ranked if pts == top_score]
        for w in sim_winners:
            win_counts[w] += (1.0 / len(sim_winners))

    return {m: round((win_counts[m] / num_simulations) * 100, 1) for m in managers}


# ==========================================
# UI SETUP & DASHBOARD RENDERING
# ==========================================
st.title("⚽ FPL Draft Rewards & Probability Dashboard")

# 1. Run the automatic background DB sync check on load
with st.spinner("Ensuring database is up to date..."):
    check_and_auto_sync()

# 2. Load the data instantly from SQLite
raw_df, last_updated, error = load_data_from_db()

# Sidebar Setup
st.sidebar.markdown(f"**Database Status:**")
if last_updated:
    st.sidebar.success(f"Last fetched: {last_updated}")
else:
    st.sidebar.warning("No database found.")

if st.sidebar.button("🔄 Force Sync Latest Points"):
    with st.spinner("Querying API..."):
        success, msg = sync_fpl_draft_db()
        if success:
            st.rerun()
        else:
            st.sidebar.error(msg)

if error:
    st.error(error)
elif raw_df is not None and not raw_df.empty:
    
    # ==========================================
    # VALIDATION: CONSIDER ONLY GWs WITH POINTS > 0
    # ==========================================
    gw_sums = raw_df.groupby("GW")["Points"].sum()
    valid_gws = gw_sums[gw_sums > 0].index.tolist()
    raw_df = raw_df[raw_df["GW"].isin(valid_gws)]
    
    max_played_gw = int(raw_df["GW"].max()) if not raw_df.empty else 0
    all_gw_cols = [f"GW{i}" for i in range(1, 39)]
    all_managers = sorted(raw_df["Teams"].unique()) if not raw_df.empty else []
    played_gw_cols = [f"GW{i}" for i in range(1, max_played_gw + 1)]

    # 1. POINTS MATRIX
    if not raw_df.empty:
        points_pivot = raw_df.pivot(index="Teams", columns="GW", values="Points").reindex(columns=range(1, 39))
    else:
        points_pivot = pd.DataFrame(index=all_managers, columns=range(1, 39))
        
    points_pivot.columns = all_gw_cols
    if played_gw_cols:
        points_pivot["Total"] = points_pivot[played_gw_cols].sum(axis=1)
        points_pivot["Average"] = (points_pivot[played_gw_cols].mean(axis=1)).round(1)
    else:
        points_pivot["Total"] = 0
        points_pivot["Average"] = 0.0

    # 2. WEEKLY PODIUM WINNERS
    winners_dict = {1: {}, 2: {}, 3: {}, 4: {}}
    for gw in range(1, 39):
        col_name = f"GW{gw}"
        if gw <= max_played_gw:
            gw_ranks = raw_df[raw_df["GW"] == gw].sort_values(by="Points", ascending=False).reset_index(drop=True)
            for pos in range(1, 5):
                winners_dict[pos][col_name] = gw_ranks.loc[pos - 1, "Teams"] if len(gw_ranks) >= pos else ""
        else:
            for pos in range(1, 5):
                winners_dict[pos][col_name] = ""

    winners_df = pd.DataFrame(winners_dict).T
    winners_df.index.name = "Winners"

    # 3. MOTM CASH CALCULATION
    motm_wins_count = {m: 0 for m in all_managers}
    motm_cash_won = {m: 0.0 for m in all_managers}

    for month, gws in GW_MONTH_MAPPING.items():
        if all(gw <= max_played_gw for gw in gws):  
            m_df = raw_df[raw_df["GW"].isin(gws)]
            if not m_df.empty:
                m_totals = m_df.groupby("Teams")["Points"].sum()
                top_pts = m_totals.max()
                month_winners = m_totals[m_totals == top_pts].index.tolist()
                prize_per_mgr = MOTM_PRIZE / len(month_winners)
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

    # 5. SUMMARY & MATRIX
    summary_data = []
    cash_matrix = pd.DataFrame(0, index=all_managers, columns=all_gw_cols)

    for manager in all_managers:
        counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for gw in range(1, max_played_gw + 1):
            col_name = f"GW{gw}"
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
            "Weekly Cash (₹)": weekly_amount,
            "MOTM Cash (₹)": int(motm_amount) if motm_amount.is_integer() else motm_amount,
            "Season End Cash (₹)": season_amount,
            "Total Cash (₹)": int(total_cash) if float(total_cash).is_integer() else total_cash
        })

    if summary_data:
        summary_df = pd.DataFrame(summary_data).set_index("Teams")
        cash_matrix["Total Weekly Cash"] = cash_matrix[played_gw_cols].sum(axis=1) if played_gw_cols else 0
    else:
        summary_df = pd.DataFrame()

    # ==========================================
    # DASHBOARD TABS
    # ==========================================
    tab_overview, tab_cash, tab_motm, tab_prob, tab_live = st.tabs([
        "📊 Points & Standings", "💰 Podium Counts & Cash Won", "👑 Manager of the Month", "🎲 Win Probabilities (%)", "🔴 Live GW Points"
    ])

    with tab_overview:
        st.subheader("📋 Points Matrix (GW1 - GW38)")
        # Expanders removed as requested
        st.dataframe(points_pivot.fillna(""), use_container_width=True)

        st.subheader("🏆 Weekly Podium Winners (1st - 4th)")
        # Expanders removed as requested
        st.dataframe(winners_df.fillna(""), use_container_width=True)

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

        st.markdown("---")
        st.subheader("💳 Weekly Cash Won per Gameweek (₹)")
        st.dataframe(cash_matrix, use_container_width=True)

    with tab_motm:
        st.subheader("👑 Manager of the Month Standings")
        selected_month = st.selectbox("Select Calendar Month", list(GW_MONTH_MAPPING.keys()))
        target_gws = GW_MONTH_MAPPING[selected_month]
        
        is_month_complete = all(gw <= max_played_gw for gw in target_gws)
        status_text = "✅ **Month Completed (Prize Awarded)**" if is_month_complete else f"⏳ **In Progress / Upcoming** (Gameweeks: {target_gws})"
        st.markdown(status_text)

        motm_filtered = raw_df[raw_df["GW"].isin(target_gws)]
        if not motm_filtered.empty:
            motm_summary = motm_filtered.groupby("Teams")["Points"].sum().reset_index().sort_values(by="Points", ascending=False).reset_index(drop=True)
            motm_summary.index += 1
            top_score = motm_summary.iloc[0]["Points"]
            current_leaders = motm_summary[motm_summary["Points"] == top_score]["Teams"].tolist()

            if is_month_complete:
                st.success(f"🎉 **Official MOTM Winner(s):** {', '.join(current_leaders)} with **{top_score}** pts (Won ₹{MOTM_PRIZE/len(current_leaders):.0f} each)!")
            else:
                st.info(f"Leader so far: **{', '.join(current_leaders)}** ({top_score} pts). Cash will be awarded after all GWs finish.")
            st.dataframe(motm_summary, use_container_width=True)
        else:
            st.info(f"No Gameweek points finalized yet for {selected_month} (GWs: {target_gws}).")

    with tab_prob:
        st.header("🎲 Monte Carlo Win Probability Projections")
        st.caption("Projections based on 5,000 simulations using each manager's historical scoring rate and variance.")
        if all_managers:
            col_seas, col_month = st.columns(2)
            with col_seas:
                st.subheader("🏆 End-of-Season Probability (GW38)")
                season_probs = run_monte_carlo_season_projections(raw_df, all_managers, max_played_gw)
                season_prob_df = pd.DataFrame([{"Teams": m, "Current Pts": int(points_pivot.loc[m, "Total"]) if "Total" in points_pivot.columns else 0, "1st Place (%)": f"{season_probs.get(m, (0,0))[0]}%", "2nd Place (%)": f"{season_probs.get(m, (0,0))[1]}%"} for m in all_managers]).sort_values(by="Current Pts", ascending=False).reset_index(drop=True)
                st.dataframe(season_prob_df, use_container_width=True, hide_index=True)
            with col_month:
                st.subheader(f"👑 MOTM Probability: {selected_month}")
                motm_probs = run_monte_carlo_motm_projections(raw_df, all_managers, target_gws, max_played_gw)
                motm_prob_df = pd.DataFrame([{"Teams": m, "Win MOTM Prob (%)": f"{motm_probs.get(m, 0)}%"} for m in all_managers]).sort_values(by="Win MOTM Prob (%)", ascending=False).reset_index(drop=True)
                st.dataframe(motm_prob_df, use_container_width=True, hide_index=True)
                
    with tab_live:
        st.subheader("🔴 Live Gameweek Points Tracker")
        st.markdown("Monitor live points. *(If the embed fails, [click here to open the tracker](https://www.anewpla.net/fpl/live/))*")
        
        # Kept the lazy-loading iframe logic intact for optimal frontend performance 
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
