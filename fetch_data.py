import sqlite3
import requests
import pandas as pd
import concurrent.futures
from datetime import datetime

LEAGUE_ID = "23942"
DB_NAME = "fpl_draft.db"
LEAGUE_DETAILS_URL = "https://draft.premierleague.com/api/league/{}/details"
ENTRY_HISTORY_URL = "https://draft.premierleague.com/api/entry/{}/history"

def sync_fpl_draft_db(league_id=LEAGUE_ID):
    print(f"[{datetime.now()}] Fetching data for league {league_id}...")
    
    res = requests.get(LEAGUE_DETAILS_URL.format(league_id))
    if res.status_code != 200:
        print(f"Error: API returned status code {res.status_code}")
        return

    league_data = res.json()
    entries = league_data.get("league_entries", [])
    if not entries:
        print("No teams found.")
        return

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
                    "Team_Name": team_name,
                    "GW": gw["event"],
                    "Points": gw["points"]
                })
        return records

    all_records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for result in executor.map(fetch_entry, entries):
            all_records.extend(result)

    if not all_records:
        print("No history records retrieved.")
        return

    df = pd.DataFrame(all_records)

    # Save to SQLite
    conn = sqlite3.connect(DB_NAME)
    df.to_sql("gw_points", conn, if_exists="replace", index=False)
    
    # Store last synced timestamp
    meta_df = pd.DataFrame([{"last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
    meta_df.to_sql("metadata", conn, if_exists="replace", index=False)
    
    conn.close()
    print(f"[{datetime.now()}] Database successfully updated in '{DB_NAME}' with {len(df)} records.")

if __name__ == "__main__":
    sync_fpl_draft_db()
