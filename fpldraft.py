import streamlit as st
import requests
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="FPL Draft Dashboard & Rewards", page_icon="⚽", layout="wide")

LEAGUE_DETAILS_URL = "https://draft.premierleague.com/api/league/{}/details"
ENTRY_HISTORY_URL = "https://draft.premierleague.com/api/entry/{}/history"

# Prize distribution structure
PRIZE_MAP = {
    1: 200,
    2: 150,
    3: 100,
    4: 50
}

# Standard FPL Gameweek to Calendar Month mapping (adjustable)
GW_MONTH_MAPPING = {
    "August": list(range(1, 4)),
    "September": list(range(4, 7)),
    "October": list(range(7, 10)),
    "November": list(range(10, 14)),
    "December": list(range(14, 20)),
    "January": list(range(20, 24)),
    "February": list(range(24, 28)),
    "March": list(range(28, 31)),
    "April": list(range(31, 35)),
    "May": list(range(35, 39))
}

@st.cache_data(ttl=900)
def load_league_data(league_id: int):
    """Fetches league managers and full Gameweek history."""
    try:
        res = requests.get(LEAGUE_DETAILS_URL.format(league_id))
        if res.status_code != 200:
            return None, f"Failed to retrieve league details (Status code: {res.status_code})"
        
        league_data = res.json()
        entries = league_data.get("league_entries", [])
        if not entries:
            return None, "No teams found in this league."

        records = []
        for entry in entries:
            entry_id = entry["entry_id"]
            manager_name = entry["player_first_name"] # First name matches sheet format
            team_name = entry["entry_name"]
            
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

        df = pd.DataFrame(records)
        return df, None
    except Exception as e:
        return None, str(e)


# --- UI Header ---
st.title("⚽ FPL Draft Rewards & Points Dashboard")

league_id = st.sidebar.text_input("FPL Draft League ID", value="23942")

if league_id:
    with st.spinner("Fetching live data from FPL Draft API..."):
        raw_df, error = load_league_data(league_id)

    if error:
        st.error(error)
    elif raw_df is not None and not raw_df.empty:
        max_played_gw = int(raw_df["GW"].max()) if not raw_df.empty else 0
        all_gw_cols = [f"GW{i}" for i in range(1, 39)]
        all_managers = sorted(raw_df["Teams"].unique())

        # ==========================================
        # 1. BUILD POINTS MATRIX (Top Sheet Table)
        # ==========================================
        points_pivot = raw_df.pivot(index="Teams", columns="GW", values="Points").reindex(columns=range(1, 39))
        points_pivot.columns = all_gw_cols
        
        # Calculate Total and Average based on completed Gameweeks
        played_gw_cols = [f"GW{i}" for i in range(1, max_played_gw + 1)]
        points_pivot["Total"] = points_pivot[played_gw_cols].sum(axis=1)
        points_pivot["Average"] = (points_pivot[played_gw_cols].mean(axis=1)).round(1)
        
        # ==========================================
        # 2. BUILD WINNERS PER GW (Middle Table)
        # ==========================================
        winners_dict = {1: {}, 2: {}, 3: {}, 4: {}}
        
        for gw in range(1, 39):
            col_name = f"GW{gw}"
            if gw <= max_played_gw:
                gw_ranks = raw_df[raw_df["GW"] == gw].sort_values(by="Points", ascending=False).reset_index(drop=True)
                for pos in range(1, 5):
                    if len(gw_ranks) >= pos:
                        winners_dict[pos][col_name] = gw_ranks.loc[pos - 1, "Teams"]
                    else:
                        winners_dict[pos][col_name] = ""
            else:
                for pos in range(1, 5):
                    winners_dict[pos][col_name] = ""

        winners_df = pd.DataFrame(winners_dict).T
        winners_df.index.name = "Winners"

        # ==========================================
        # 3. BUILD PRIZE & CASH SUMMARY
        # ==========================================
        # Summary counts (1st, 2nd, 3rd, 4th, Total Podiums, Total Cash)
        summary_data = []
        # Matrix of cash won per GW
        cash_matrix = pd.DataFrame(0, index=all_managers, columns=all_gw_cols)

        for manager in all_managers:
            counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for gw in range(1, max_played_gw + 1):
                col_name = f"GW{gw}"
                for pos in [1, 2, 3, 4]:
                    if winners_dict[pos].get(col_name) == manager:
                        counts[pos] += 1
                        cash_matrix.loc[manager, col_name] = PRIZE_MAP[pos]
            
            total_wins = sum(counts.values())
            total_amount = sum(counts[p] * PRIZE_MAP[p] for p in PRIZE_MAP)
            
            summary_data.append({
                "Teams": manager,
                "1st": counts[1],
                "2nd": counts[2],
                "3rd": counts[3],
                "4th": counts[4],
                "Total Podiums": total_wins,
                "Total Amount (₹/$)": total_amount
            })

        summary_df = pd.DataFrame(summary_data).set_index("Teams")
        cash_matrix["Total Amount"] = cash_matrix[played_gw_cols].sum(axis=1)

        # ==========================================
        # DASHBOARD TABS
        # ==========================================
        tab_overview, tab_cash, tab_motm = st.tabs([
            "📊 Points & Standings",
            "💰 Prize Distribution & Earnings",
            "👑 Manager of the Month"
        ])

        # --- TAB 1: Points & Weekly Winners ---
        with tab_overview:
            st.subheader("📋 Points Matrix (GW1 - GW38)")
            st.dataframe(points_pivot.fillna(""), use_container_width=True)

            st.subheader("🏆 Weekly Podium Winners (1st - 4th)")
            st.dataframe(winners_df.fillna(""), use_container_width=True)

        # --- TAB 2: Cash Breakdown & Distribution ---
        with tab_cash:
            col_left, col_right = st.columns([1, 1.5])
            
            with col_left:
                st.subheader("🎖️ Podium Counts & Total Cash Won")
                st.dataframe(summary_df.sort_values(by="Total Amount (₹/$)", ascending=False), use_container_width=True)

            with col_right:
                st.subheader("💵 Prize Payout Rules")
                prize_rule_df = pd.DataFrame({
                    "Position": ["1st Place", "2nd Place", "3rd Place", "4th Place"],
                    "Winning Amount": [f"{PRIZE_MAP[1]}", f"{PRIZE_MAP[2]}", f"{PRIZE_MAP[3]}", f"{PRIZE_MAP[4]}"]
                })
                st.dataframe(prize_rule_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("💳 Cash Won by Gameweek (GW1 - GW38)")
            st.dataframe(cash_matrix, use_container_width=True)

        # --- TAB 3: Manager of the Month ---
        with tab_motm:
            st.subheader("👑 Manager of the Month Standings")
            
            mode = st.radio("Calculation Mode", ["Calendar Month Presets", "Custom Gameweek Range"], horizontal=True)
            
            if mode == "Calendar Month Presets":
                selected_month = st.selectbox("Select Month", list(GW_MONTH_MAPPING.keys()))
                target_gws = GW_MONTH_MAPPING[selected_month]
                st.caption(f"Gameweeks included: {target_gws}")
            else:
                target_gws = st.slider(
                    "Select Gameweek Range", 
                    min_value=1, 
                    max_value=38, 
                    value=(1, min(max_played_gw if max_played_gw > 0 else 4, 38))
                )
                target_gws = list(range(target_gws[0], target_gws[1] + 1))

            # Calculate points for chosen range
            motm_filtered = raw_df[raw_df["GW"].isin(target_gws)]
            
            if not motm_filtered.empty:
                motm_summary = (
                    motm_filtered.groupby("Teams")["Points"]
                    .sum()
                    .reset_index()
                    .sort_values(by="Points", ascending=False)
                    .reset_index(drop=True)
                )
                motm_summary.index += 1 # 1-based rank

                top_score = motm_summary.iloc[0]["Points"]
                winners = motm_summary[motm_summary["Points"] == top_score]["Teams"].tolist()
                
                st.success(f"🎉 **Manager of the Month Winner(s):** {', '.join(winners)} with **{top_score}** points!")
                st.dataframe(motm_summary, use_container_width=True)
            else:
                st.info("No Gameweek data available for the selected range yet.")
else:
    st.info("Please enter a valid Draft League ID in the sidebar.")
