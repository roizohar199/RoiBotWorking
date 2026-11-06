"""Build the two Pine variants from one policy template; no trading actions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "אינדיקטוריםTRADING VIEW"
VARIANTS = (
    ("instant_breakout_rsi_vwap_5sec_regime.pine", "NQ Breakout - 5sec - Regime", "false", "5sec-regime"),
    ("instant_breakout_rsi_vwap_5sec_reversed_regime.pine", "NQ Reversed - 5sec - Regime", "true", "5sec-reversed-regime"),
)


def build(destination=DESTINATION):
    source = (ROOT / "pine_templates" / "regime_5sec.pine.in").read_text(encoding="utf-8")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, title, reversed_mode, tag in VARIANTS:
        content = source.replace("__TITLE__", title).replace("__REVERSED__", reversed_mode).replace("__TAG__", tag)
        assert "__" not in content
        path = destination / filename
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for path in build():
        print(path.name)
