import streamlit as st
import requests
import pandas as pd

# --- FPL Draft API Endpoints ---
LEAGUE_URL = "https://draft.premierleague.com/api/league/{}/details"
ENTRY_HISTORY_URL = "https://draft.premierleague.com/api/entry/{}/history"

st.set_page_config(page_title="FPL Draft Rewards", page_icon="🏆", layout="wide")

@st.cache_data(ttl=3600) # Cache data for 1 hour to avoid spamming the API
def fetch_league_data(league_id):
    """Fetch league entries and compile weekly history for all managers."""
    league_res = requests.get(LEAGUE_URL.format(league_id))
    
    if league_res.status_code != 200:
        return None, "League not found or API is down."
    
    league_data = league_res.json()
    entries = league_data.get("league_entries", [])
    
    if not entries:
        return None, "No managers found in this league."

    all_history = []
    
    for entry in entries:
        entry_id = entry["entry_id"]
        manager_name = f"{entry['player_first_name']} {entry['player_last_name']}"
        team_name = entry["entry_name"]
        
        # Fetch history for each manager
        history_res = requests.get(ENTRY_HISTORY_URL.format(entry_id))
        if history_res.status_code == 200:
            history_data = history_res.json().get("history", [])
            for gw in history_data:
                all_history.append({
                    "Manager": manager_name,
                    "Team": team_name,
                    "GW": gw["event"],
                    "Points": gw["points"],
                    "Total Points": gw["total_points"]
                })
                
    df = pd.DataFrame(all_history)
    return df, None

# --- UI Setup ---
st.title("🏆 FPL Draft League Rewards Dashboard")

# Sidebar for config
st.sidebar.header("League Configuration")
league_id = st.sidebar.text_input("Enter Draft League ID", value="")

if league_id:
    with st.spinner("Fetching data from FPL Draft API..."):
        df, error = fetch_league_data(league_id)
        
    if error:
        st.error(error)
    elif df is not None and not df.empty:
        max_gw = df["GW"].max()
        
        # Create Tabs
        tab1, tab2 = st.tabs(["💰 Weekly Cash Prizes", "👑 Manager of the Month"])
        
        # --- TAB 1: Weekly Prizes ---
        with tab1:
            st.header("Weekly Points Leaderboard")
            selected_gw = st.selectbox("Select Gameweek", range(1, max_gw + 1), index=max_gw-1)
            
            # Filter and sort
            gw_df = df[df["GW"] == selected_gw].sort_values(by="Points", ascending=False).reset_index(drop=True)
            gw_df.index += 1 # 1-based index
            
            # Identify top 4
            top_4 = gw_df.head(4).copy()
            top_4["Prize Rank"] = ["🥇 1st Place", "🥈 2nd Place", "🥉 3rd Place", "🏅 4th Place"]
            
            st.subheader(f"Top 4 Managers for GW {selected_gw}")
            st.dataframe(
                top_4[["Prize Rank", "Manager", "Team", "Points"]],
                use_container_width=True,
                hide_index=True
            )
            
            with st.expander("Show Full Gameweek Standings"):
                st.dataframe(gw_df[["Manager", "Team", "Points"]], use_container_width=True)

        # --- TAB 2: Manager of the Month ---
        with tab2:
            st.header("Manager of the Month (MOTM)")
            st.markdown("Select the range of Gameweeks that correspond to the calendar month.")
            
            # Dual slider to select GW range for a specific month
            gw_range = st.slider("Select Gameweek Range", min_value=1, max_value=int(max_gw), value=(1, min(4, max_gw)))
            
            # Filter for the range and sum points
            motm_df = df[(df["GW"] >= gw_range[0]) & (df["GW"] <= gw_range[1])]
            motm_grouped = motm_df.groupby(["Manager", "Team"])["Points"].sum().reset_index()
            motm_grouped = motm_grouped.sort_values(by="Points", ascending=False).reset_index(drop=True)
            motm_grouped.index += 1
            
            if not motm_grouped.empty:
                winner = motm_grouped.iloc[0]
                
                st.success(f"### 👑 MOTM Winner: {winner['Manager']} ({winner['Team']}) with {winner['Points']} points!")
                st.balloons()
                
                st.subheader(f"Points Breakdown (GW {gw_range[0]} to GW {gw_range[1]})")
                st.dataframe(motm_grouped, use_container_width=True)
else:
    st.info("👈 Please enter your Draft League ID in the sidebar to get started.")
    st.markdown("""
    **How to find your League ID:**
    1. Go to the FPL Draft website and click on your league.
    2. Look at the URL in your browser.
    3. It will look like `https://draft.premierleague.com/league/XXXXX/details` where `XXXXX` is your League ID.
    """)
