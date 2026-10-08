
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("BlossomTools")

@mcp.tool()
def get_federal_holidays(year: int = 2026) -> str:
    """Fetch US federal holidays to calculate bank processing times."""
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/US"
    try:
        r = requests.get(url, timeout=3)
        return "\n".join([f"{h['date']}: {h['name']}" for h in r.json()])
    except Exception:
        return "Holiday data currently unavailable."

if __name__ == "__main__":
    mcp.run()