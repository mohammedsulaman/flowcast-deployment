import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

os.environ["FLOWCAST_DEPLOYMENT_MODE"] = "true"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Execute the existing dashboard entry point.
dashboard = ROOT / "dashboard" / "app.py"

code = compile(
    dashboard.read_text(encoding="utf-8-sig"),
    str(dashboard),
    "exec"
)

exec(
    code,
    {
        "__name__": "__main__",
        "__file__": str(dashboard),
        "__package__": "dashboard"
    }
)
