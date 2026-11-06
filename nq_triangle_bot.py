"""
NQ triangle bot - screen based, no API.
Detects a new configured SELL/BUY triangle on the chart and clicks Sell/Buy.
Run on a DEMO account first.

Install:  pip install mss pyautogui numpy keyboard uiautomation
One-click mode: enable One-click in TradingView and select the preset on chart buttons.
Setup:    python nq_triangle_bot.py calibrate   (F8 positions only in one-click mode)
Buttons:  python nq_triangle_bot.py calibrate-buttons (two F8 presses, no clicks)
Colours:  python nq_triangle_bot.py calibrate-colors (two F8 presses, keeps area/buttons)
Run:      python nq_triangle_bot.py run          (ESC = emergency stop)
Test:     python nq_triangle_bot.py test         (detect only, no clicks)
Account:  python nq_triangle_bot.py check-position (read only; keep Positions open)
Area only: python nq_triangle_bot.py calibrate-region (preserves colours and buttons)
Optional ticket mode (use_ticket=true):
          python nq_triangle_bot.py calibrate-ticket (manual F8, no order clicks)
          python nq_triangle_bot.py check-ticket     (reads ticket markers, no clicks)
"""
import sys, time, json, os
import argparse
import struct
import zlib
from contextlib import contextmanager, nullcontext
from pathlib import Path
import numpy as np
import mss
import pyautogui
import keyboard
from signal_tracker import MarkerTracker, choose_signal, find_markers
from bot_journal import RunJournal
from position_guard import (AccountMonitor, BotInstanceLock, PositionGate,
                            configuration_errors, account_status, signal_scan_region)

CFG_FILE = "nq_bot_config.json"
pyautogui.PAUSE = 0          # no artificial delay between actions
pyautogui.FAILSAFE = True    # mouse to top-left corner = abort

DEFAULTS = {
    "region": None,          # {"left","top","width","height"} around the LAST candle(s)
    "red_rgb": None,         # sampled colour of the red triangle
    "green_rgb": None,       # sampled colour of the green triangle
    "tolerance": 25,         # per-channel colour tolerance
    "min_pixels": 12,        # min matching pixels to count as a triangle
    "buy_btn": None,         # [x, y] of the BUY / Long button
    "sell_btn": None,        # [x, y] of the SELL / Short button
    "use_ticket": False,     # False: one chart click; exits come from the TradingView preset
    "confirm_btn": None,     # [x, y] of the confirm button inside the order ticket
    "confirm_buy_rgb": None, # solid background sampled on the final BUY button
    "confirm_sell_rgb": None,# same screen point on the final SELL button
    "tp_toggle": None,       # centre of the TP switch, calibrated while ON
    "sl_toggle": None,       # centre of the SL switch, calibrated while ON
    "tp_enabled_rgb": None,
    "sl_enabled_rgb": None,
    "ticket_color_tolerance": 35, # allow small hover/background shade changes
    "ticket_timeout_ms": 1500,  # max wait for the ticket to appear (it is detected automatically)
    "ticket_settle_ms": 60,     # tiny pause after the ticket appears so it finishes rendering
    "fill_sltp": False,      # ticket mode only; ignored for one-click entries
    "tp_field": None,        # [x, y] of the Take Profit input in the order ticket
    "sl_field": None,        # [x, y] of the Stop Loss input in the order ticket
    "tp_value": "5",         # NQ: 5 points = 20 ticks = $100 (use the unit the field uses)
    "sl_value": "5",         # same for SL
    "cooldown_sec": 5,       # min seconds between two entries
    "poll_ms": 5,            # target capture-start interval (processing included)
    "scan_bound_chart": False, # automatically scan the bound 1-second price pane
    "position_guard": {},   # required bound account/page; unreadable = no entry
}


def load():
    if not os.path.exists(CFG_FILE):
        return dict(DEFAULTS)
    with open(CFG_FILE) as f:
        return {**DEFAULTS, **json.load(f)}


def save(cfg):
    with open(CFG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def countdown(msg, secs=0):
    # Consume the previous F8 hold, then sample this point immediately on the
    # NEXT press. A fixed post-press delay can let a moving candle leave it.
    while keyboard.is_pressed("f8"):
        time.sleep(0.01)
    print(f"\n{msg}\n  -> move the mouse there and press F8")
    keyboard.wait("f8")
    pos = pyautogui.position()
    print(f"  saved ({pos.x}, {pos.y})")
    return pos


def signal_color_errors(cfg):
    """Reject neutral samples and overlapping BUY/SELL colour tolerances."""
    errors = []
    colours = []
    for key, label in (("red_rgb", "SELL"), ("green_rgb", "BUY")):
        rgb = cfg.get(key)
        if (not isinstance(rgb, (list, tuple)) or len(rgb) != 3
                or any(not isinstance(value, int) or not 0 <= value <= 255 for value in rgb)):
            errors.append(f"{label}: missing or invalid RGB colour")
            continue
        colours.append(rgb)
        if max(rgb) < 80 or max(rgb) - min(rgb) < 60:
            errors.append(f"{label}: RGB {rgb} is dark/grey; sample the coloured triangle, away from labels")
    if len(colours) == 2 and all(abs(a - b) <= 2 * cfg["tolerance"]
                                 for a, b in zip(*colours)):
        errors.append("BUY and SELL colour ranges overlap; they cannot be distinguished reliably")
    return errors


def sample_signal_color(point):
    """Use the dominant coloured pixels near F8, ignoring a grey crosshair."""
    with mss.MSS() as capture:
        patch = np.asarray(capture.grab({"left": int(point.x) - 6, "top": int(point.y) - 6,
                                       "width": 13, "height": 13}))
    rgb = patch[:, :, 2::-1].reshape(-1, 3)
    saturated = rgb[(rgb.max(axis=1) >= 80) & (np.ptp(rgb, axis=1) >= 60)]
    if len(saturated) < 5:
        return None
    colours, counts = np.unique(saturated, axis=0, return_counts=True)
    index = int(counts.argmax())
    return colours[index].tolist() if counts[index] >= 5 else None


def read_signal_colors(cfg):
    sampled = dict(cfg)
    for key, label in (("red_rgb", "SELL (orange)"), ("green_rgb", "BUY (cyan)")):
        point = countdown(f"Hover over the CENTER of a {label} TRIANGLE")
        colour = sample_signal_color(point)
        if colour is None:
            print("NOT SAVED: no clear coloured triangle near the pointer; existing settings preserved.")
            return False
        sampled[key] = colour
        print(f"  {label} RGB: {colour}")
    errors = signal_color_errors(sampled)
    if errors:
        print("NOT SAVED:", "; ".join(errors))
        return False
    cfg.update(red_rgb=sampled["red_rgb"], green_rgb=sampled["green_rgb"])
    return True


def calibrate_colors():
    cfg = load()
    if read_signal_colors(cfg):
        save(cfg)
        print("Signal colours saved. Scan area, buttons and other settings preserved.")


def calibrate():
    cfg = load()
    # A new layout invalidates the separately calibrated ticket markers.
    for key in ("confirm_buy_rgb", "confirm_sell_rgb", "tp_toggle", "sl_toggle",
                "tp_enabled_rgb", "sl_enabled_rgb"):
        cfg[key] = None
    p1 = countdown("Hover over the TOP-LEFT corner of the scan area (right edge of the chart, around the last 2 candles)")
    p2 = countdown("Hover over the BOTTOM-RIGHT corner of the scan area")
    cfg["region"] = {
        "left": min(p1.x, p2.x), "top": min(p1.y, p2.y),
        "width": abs(p2.x - p1.x), "height": abs(p2.y - p1.y),
    }
    if not read_signal_colors(cfg):
        return
    pb = countdown("Hover over the BUY (long) button")
    cfg["buy_btn"] = [pb.x, pb.y]
    ps = countdown("Hover over the SELL (short) button")
    cfg["sell_btn"] = [ps.x, ps.y]
    if cfg["use_ticket"]:
        print("\nOpen the order ticket manually with One-click DISABLED. Do NOT submit an order.")
        print("Make sure the Take Profit and Stop Loss rows are enabled/ticked.")
        pt = countdown("Hover over the TAKE PROFIT input field in the ticket", 10)
        cfg["tp_field"] = [pt.x, pt.y]
        pl = countdown("Hover over the STOP LOSS input field in the ticket")
        cfg["sl_field"] = [pl.x, pl.y]
        pc = countdown("Hover over the final confirm (Buy) button inside the ticket")
        cfg["confirm_btn"] = [pc.x, pc.y]
        print("\nClose the ticket without sending the order. Run calibrate-ticket before run.")
        print("tp_value/sl_value must match the ticket's input units.")
    else:
        print("\nOne-click mode: no ticket calibration or field filling is used.")
        print("Select the desired TP/SL preset in TradingView on the chart Buy/Sell buttons.")
    warn = signal_color_errors(cfg)
    r = cfg["region"]
    if r["width"] < 20 or r["height"] < 20:
        warn.append("scan region is very small - it should cover the last candle(s), roughly 60x200 px or more")
    if abs(cfg["buy_btn"][0] - cfg["sell_btn"][0]) > 400 or abs(cfg["buy_btn"][1] - cfg["sell_btn"][1]) > 400:
        warn.append("Buy and Sell buttons are far apart - usually they are next to each other on the chart")
    if cfg["use_ticket"] and abs(cfg["tp_field"][1] - cfg["sl_field"][1]) > 400:
        warn.append("TP and SL fields are far apart - they should be in the same order ticket")
    if warn:
        print("\nNOT SAVED - invalid calibration; previous settings preserved:")
        for w in warn:
            print("  -", w)
        return
    else:
        print("\nCalibration looks reasonable. Now run: py nq_triangle_bot.py test")
    save(cfg)
    print("\nSaved to", CFG_FILE)
    print(json.dumps(cfg, indent=2))


def calibrate_region():
    """Update only the screen scan area; never open or submit an order."""
    cfg = load()
    print("Region-only calibration: colours and order fields are preserved.")
    print("Use a narrow vertical strip around the LAST candle(s), including triangle width.")
    print("Exclude old signals, chart labels, buttons and the price scale.")
    p1 = countdown("Hover over the TOP-LEFT corner of the scan area")
    p2 = countdown("Hover over the BOTTOM-RIGHT corner of the scan area")
    region = {
        "left": min(p1.x, p2.x), "top": min(p1.y, p2.y),
        "width": abs(p2.x - p1.x), "height": abs(p2.y - p1.y),
    }
    if not 20 <= region["width"] <= 120 or region["height"] < 100:
        print("NOT SAVED: use a strip 20-120 pixels wide and at least 100 pixels tall.")
        print("Received:", json.dumps(region))
        print("Run calibrate-region again. Existing settings were preserved.")
        return
    cfg["region"] = region
    cfg["scan_bound_chart"] = False
    save(cfg)
    print("Saved scan area:", json.dumps(region))
    print("Next: py nq_triangle_bot.py test (no clicks). ESC stops the test.")


def calibrate_buttons():
    """Recalibrate only chart Buy/Sell coordinates using manual F8 presses."""
    cfg = load()
    print("Show the chart buttons with the desired preset selected.")
    print("HOVER ONLY and press F8. Do NOT click Buy/Sell while One-click is enabled.")
    buy = countdown("Hover over the CENTRE of the chart BUY button")
    sell = countdown("Hover over the CENTRE of the chart SELL button")
    distance = abs(buy.x - sell.x) + abs(buy.y - sell.y)
    if distance < 20 or max(abs(buy.x - sell.x), abs(buy.y - sell.y)) > 400:
        print("NOT SAVED: chart Buy/Sell points should be distinct and near each other.")
        return
    cfg["buy_btn"] = [buy.x, buy.y]
    cfg["sell_btn"] = [sell.x, sell.y]
    save(cfg)
    print("Chart buttons saved. Scan area, colours and other settings were preserved.")


def count_color(img, rgb, tol):
    # img is BGRA from mss
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    m = (abs(r - rgb[0]) <= tol) & (abs(g - rgb[1]) <= tol) & (abs(b - rgb[2]) <= tol)
    return int(m.sum())


def px(x, y):
    with mss.MSS() as s:
        a = np.array(s.grab({"left": int(x), "top": int(y), "width": 1, "height": 1}))
    return int(a[0, 0, 2]), int(a[0, 0, 1]), int(a[0, 0, 0])


def log(msg):
    print(time.strftime("%H:%M:%S"), f"{time.time() % 1:.3f}"[1:], msg, flush=True)


def save_test_observation(img, cfg, markers, selected, messages, slot, folder=None):
    """Save already-captured TEST pixels locally, never take another screenshot.

    Keep one baseline and a ring of 20 observations. This runs only in TEST,
    after the decision, and adds no disk I/O to live entry handling.
    """
    folder = Path(folder) if folder is not None else Path(__file__).with_name("nq_bot_diagnostics")
    folder.mkdir(parents=True, exist_ok=True)
    stem = "baseline" if slot == 0 else f"observation-{(slot - 1) % 20:02d}"
    height, width = img.shape[:2]
    rgb = img[:, :, 2::-1]
    scanlines = b"".join(b"\x00" + row.tobytes() for row in rgb)

    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))
    path = folder / f"{stem}.png"
    path.write_bytes(png)
    metadata = {"observed_unix": time.time(), "scan_region": cfg["region"],
                "sell_rgb": cfg["red_rgb"], "buy_rgb": cfg["green_rgb"],
                "visible_triangles": [vars(marker) for marker in markers],
                "selected": vars(selected) if selected is not None else None,
                "messages": messages}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return str(path)


def ticket_required(cfg):
    keys = ["confirm_btn", "confirm_buy_rgb", "confirm_sell_rgb"]
    if cfg["fill_sltp"]:
        keys += ["tp_field", "sl_field", "tp_toggle", "sl_toggle",
                 "tp_enabled_rgb", "sl_enabled_rgb"]
    return keys


def ticket_matches(cfg, side):
    """Check calibrated visual markers, not order execution or input values."""
    if any(not cfg.get(key) for key in ticket_required(cfg)):
        return False
    markers = [(cfg["confirm_btn"], cfg[f"confirm_{side}_rgb"])]
    if cfg["fill_sltp"]:
        markers += [(cfg["tp_toggle"], cfg["tp_enabled_rgb"]),
                    (cfg["sl_toggle"], cfg["sl_enabled_rgb"])]
    tolerance = cfg["ticket_color_tolerance"]
    return all(max(abs(a - b) for a, b in zip(px(*point), expected)) <= tolerance
               for point, expected in markers)


def calibrate_ticket():
    """Only observe manually selected positions; never click a trading control."""
    cfg = load()
    print("Open the Paper Trading MARKET ticket manually with the preset applied.")
    print("Set Units=1, TP=20 ticks and SL=20 ticks. Both switches must be ON.")
    print("Keep the same window position/size. NEVER click the final Buy/Sell button.")
    for name, label in (("tp", "TAKE PROFIT"), ("sl", "STOP LOSS")):
        point = countdown(f"Hover over the CENTRE of the {label} switch while it is ON")
        colour = list(px(point.x, point.y))
        if max(colour) < 180:
            print("NOT SAVED: switch sample is too dark; enable it and try again.")
            return
        cfg[f"{name}_toggle"] = [point.x, point.y]
        cfg[f"{name}_enabled_rgb"] = colour
    for name, label in (("tp", "TAKE PROFIT"), ("sl", "STOP LOSS")):
        point = countdown(f"Hover INSIDE the LEFT {label} numeric input (ticks)")
        cfg[f"{name}_field"] = [point.x, point.y]
    point = countdown("Select BUY at the TOP of the ticket, then hover on a plain BLUE part of the BOTTOM confirmation button, away from text")
    cfg["confirm_btn"] = [point.x, point.y]
    cfg["confirm_buy_rgb"] = list(px(point.x, point.y))
    countdown("Select SELL at the TOP of the ticket, then hover on the SAME plain part of the BOTTOM confirmation button (now RED)")
    cfg["confirm_sell_rgb"] = list(px(*cfg["confirm_btn"]))
    buy, sell = cfg["confirm_buy_rgb"], cfg["confirm_sell_rgb"]
    if not (buy[2] > max(buy[0], buy[1]) + 40 and
            sell[0] > max(sell[1], sell[2]) + 40):
        print("NOT SAVED: expected BLUE Buy / RED Sell button backgrounds. Try again, away from text.")
        return
    # The SELL side is still open at the end of this manual calibration.
    if not ticket_matches(cfg, "sell"):
        print("NOT SAVED: ticket markers do not match; keep both switches ON and the layout fixed.")
        return
    save(cfg)
    print("Ticket calibration saved. Next: py nq_triangle_bot.py check-ticket")
    print("No order was clicked. Detection colours, scan area and chart buttons were preserved.")


def check_ticket():
    """Read-only check of a ticket the user has already opened."""
    cfg = load()
    missing = [key for key in ticket_required(cfg) if not cfg.get(key)]
    if missing:
        print("Missing ticket calibration:", ", ".join(missing))
        print("Run: py nq_triangle_bot.py calibrate-ticket")
        return
    for side in ("buy", "sell"):
        if ticket_matches(cfg, side):
            print(f"PASS: {side.upper()} button and enabled TP/SL markers match. No clicks.")
            print("This checks visual markers only. Verify Paper Trading, quantity and tick values yourself.")
            return
    print("FAIL: ticket markers do not match. No clicks. Keep run stopped.")
    print("Confirm point:", cfg["confirm_btn"], "actual RGB:", px(*cfg["confirm_btn"]))
    if cfg["fill_sltp"]:
        for name in ("tp", "sl"):
            print(name.upper(), "switch actual RGB:", px(*cfg[f"{name}_toggle"]),
                  "expected ON:", cfg[f"{name}_enabled_rgb"])


def _enter_unchecked(cfg, side, signal_seen_at=None, before_confirm=None):
    """Click the chart once, or handle a calibrated ticket when use_ticket is True."""
    mod = "command" if sys.platform == "darwin" else "ctrl"
    t0 = time.monotonic()
    use_ticket = cfg["use_ticket"]
    if use_ticket:
        if any(not cfg.get(key) for key in ticket_required(cfg)):
            log("ABORT: run calibrate-ticket first; no chart button was clicked")
            return False
        if ticket_matches(cfg, "buy") or ticket_matches(cfg, "sell"):
            log("ABORT: a recognized order ticket is already open; close it before run")
            return False
    if keyboard.is_pressed("esc"):
        return False
    # 1) click the chart: immediate order in one-click mode, otherwise open a ticket.
    pyautogui.click(*(cfg["buy_btn"] if side == "buy" else cfg["sell_btn"]))
    if not use_ticket:
        clicked_at = time.monotonic()
        observed = (f"; first observation to click {(clicked_at - signal_seen_at) * 1000:.1f}ms"
                    if signal_seen_at is not None else "")
        log(f"{side.upper()} chart button clicked once in {clicked_at - t0:.3f}s{observed}; "
            "TradingView preset/account/execution not verified")
        return True
    # 2) require the correct side's button AND enabled TP/SL switch markers.
    deadline = time.monotonic() + cfg["ticket_timeout_ms"] / 1000
    while not ticket_matches(cfg, side):
        if keyboard.is_pressed("esc"):
            log("ABORT: ESC pressed; confirmation was not clicked")
            return False
        if time.monotonic() > deadline:
            log("ABORT: ticket was NOT RECOGNIZED at calibrated positions, or TP/SL are OFF; confirmation was not clicked")
            return False
        time.sleep(0.01)
    time.sleep(cfg["ticket_settle_ms"] / 1000)
    # 2) fill TP / SL ($100 each)
    if cfg["fill_sltp"]:
        for field, val in ((cfg["tp_field"], cfg["tp_value"]), (cfg["sl_field"], cfg["sl_value"])):
            if keyboard.is_pressed("esc") or not ticket_matches(cfg, side):
                log("ABORT: ticket changed or ESC pressed; confirmation was not clicked")
                return False
            pyautogui.click(*field)
            pyautogui.hotkey(mod, "a")
            pyautogui.write(str(val), interval=0)
        pyautogui.press("tab")
    # 3) confirm
    if keyboard.is_pressed("esc") or not ticket_matches(cfg, side):
        log("ABORT: ticket changed or TP/SL are OFF; confirmation was not clicked")
        return False
    if before_confirm is not None and not before_confirm():
        log("ABORT: account is no longer verified flat; confirmation was not clicked")
        return False
    pyautogui.click(*cfg["confirm_btn"])
    log(f"{side.upper()} confirmation clicked after filling SL={cfg['sl_value']} TP={cfg['tp_value']} in {time.monotonic() - t0:.2f}s; execution not verified")
    return True


def enter(cfg, side, signal_seen_at=None, guard=None):
    """All RUN entries must reserve a verified-flat account before any click."""
    if guard is None:
        log("ABORT: position guard is required; no click")
        return False
    if keyboard.is_pressed("esc"):
        return False
    if not guard.claim():
        log(guard.reason + "; signal discarded")
        return None
    # A failure/exception deliberately keeps the durable reservation. Never
    # assume that an interrupted click or a rejected order means FLAT.
    return _enter_unchecked(guard.entry_config(cfg), side, signal_seen_at, before_confirm=guard.still_flat)


@contextmanager
def position_session(cfg, dry, reset_on_start=False):
    # Keep the same OS lock across reset AND the ensuing run. TEST reset is
    # simulated but also excludes RUN so its observations stay reviewable.
    with (nullcontext() if dry and not reset_on_start else BotInstanceLock()):
        with AccountMonitor(cfg) as monitor:
            if reset_on_start:
                print("STARTUP RESET: verifying same account, FLAT, zero orders and quantity 1.")
                try:
                    gate = PositionGate(cfg, monitor, dry=dry)
                    gate.reset()
                except RuntimeError as error:
                    raise StartupResetError(str(error)) from error
                print("TEST: reset simulated; saved position lock unchanged." if dry else
                      f"STARTUP RESET VERIFIED: {gate.broker}/{gate.account}; reservation cleared. Waiting for a new signal.")
            else:
                gate = PositionGate(cfg, monitor, dry=dry)
            yield gate


class StartupResetError(RuntimeError):
    """Expected refusal: no reset and no entry loop after failed verification."""


def check_position(reset=False):
    cfg = load()
    errors = configuration_errors(cfg)
    if errors:
        print("\n".join(errors))
        return
    print("READ ONLY: no order clicks. Locating the bound account pane across the desktop; keep Positions selected.")
    # Read-only diagnostics may run while RUN holds its lock. A manual reset
    # may not: resetting an in-flight reservation from another process is unsafe.
    with (BotInstanceLock() if reset else nullcontext()):
        with AccountMonitor(cfg) as monitor:
            gate = PositionGate(cfg, monitor, dry=not reset)
            for _ in range(8):
                view = monitor.fresh()
                allowed = gate.observe(view)
                print(json.dumps(dict(position=view.position, quantity=view.quantity,
                                      account_status=account_status(view), positions_bounds=view.positions_bounds,
                                      realized_pnl=view.realized_pnl, release_evidence=gate.release_evidence,
                                      entry_observed_at=view.entry_observed_at,
                                      working_orders=view.working_orders, foreground=view.foreground,
                                      entry_surface_clear=view.entry_surface_clear, surface_detail=view.surface_detail,
                                      page_ok=view.page_ok, guard_phase=gate.phase,
                                      entry_allowed=allowed, reason=gate.reason,
                                      read_detail=view.reason), ensure_ascii=False))
                if keyboard.is_pressed("esc"):
                    return
            if reset:
                gate.reset()
                print("Position reservation reset after verified flat observations. No clicks.")


def run(dry=False, strategy_tag=None, test_seconds=None, reset_on_start=False):
    if test_seconds is not None and not dry:
        raise ValueError("A timed diagnostic is supported only in TEST mode")
    cfg = load()
    required = ["region", "red_rgb", "green_rgb"]
    if not dry:
        required += ["buy_btn", "sell_btn"]
        if cfg["use_ticket"]:
            required += ticket_required(cfg)
    for k in required:
        if not cfg[k]:
            print("Missing", k, "- run calibrate-ticket for ticket markers; calibrate-region for scan area")
            return
    errors = signal_color_errors(cfg)
    errors += configuration_errors(cfg)
    if errors:
        print("ABORT: invalid calibration/account binding; no screen scan or click was started.")
        for error in errors:
            print("  -", error)
        print("Run: py nq_triangle_bot.py calibrate-colors")
        return
    print("Running", "(TEST - no clicks)" if dry else "(LIVE on whatever account is open)",
          "- press ESC to stop")
    if not cfg["use_ticket"]:
        print("Mode: ONE-CLICK. TradingView must have One-click enabled and the TP/SL preset selected.")
        print("The bot does not fill ticket fields or verify the selected preset or trade execution.")
    else:
        print("Mode: ORDER TICKET with calibrated visual markers.")
    print(f"Cooldown: {cfg['cooldown_sec']} seconds.")
    print("POSITION GUARD: max 1 contract; open/unknown/pending = skip BOTH directions.")
    print("Keep Positions selected; order controls and scan area must be uncovered. Keyboard focus may be elsewhere.")
    print(f"Target scan interval: {cfg['poll_ms']}ms (processing included); two-capture confirmation retained.")
    print("Tracking separate markers. Existing startup markers are ignored (0.3s warmup).")
    print("Shape check: narrow tip, filled triangular body and correct direction required.")
    print("Full chart: existing triangles are baselined; only new rightmost triangles are eligible." if cfg["scan_bound_chart"] else "Keep the scan area around the LAST candles; TEST prints reasons for skipped signals.")
    print("Scan area: bound 1-second chart (existing triangles ignored at startup)" if cfg["scan_bound_chart"] else f"Scan area: {cfg['region']}")
    print(f"Signal RGB: SELL={cfg['red_rgb']} BUY={cfg['green_rgb']}")
    if dry:
        print("TEST: SCAN lines are visibility status, not new signals. Local evidence is saved for new candidates.")
    tracker = MarkerTracker()
    last_trade = None
    next_status = None
    last_guard_reason = None
    test_observation_slot = 0
    active_region = None
    poll_interval = max(0, cfg["poll_ms"]) / 1000
    session = (position_session(cfg, dry, reset_on_start=True) if reset_on_start else
               position_session(cfg, dry))
    with session as guard, RunJournal(strategy_tag, dry, cfg) as journal, mss.MSS() as sct:
        if reset_on_start:
            journal.record("startup_position_reset", verified=True, simulated=dry,
                           account=guard.account, broker=guard.broker, phase=guard.phase)
        test_deadline = time.monotonic() + test_seconds if test_seconds is not None else None
        print(f"Declared strategy: {journal.strategy_tag}; journal: {journal.path}")
        if journal.strategy_tag == "unlabeled":
            print("Use --strategy LABEL to attribute this session; the chart version is not auto-detected.")
        while not keyboard.is_pressed("esc"):
            if test_deadline is not None and time.monotonic() >= test_deadline:
                break
            frame_started = time.monotonic()
            chart_bounds = guard.monitor.latest.chart_bounds
            scan_region = signal_scan_region(cfg, chart_bounds)
            if scan_region is None:
                guard.observe(guard.monitor.latest)
                if next_status is None or time.monotonic() >= next_status:
                    detail = guard.monitor.latest.reason
                    journal.record("scan_unavailable", reason=detail)
                    log("WAIT SCAN: " + detail)
                    next_status = time.monotonic() + 5
                time.sleep(.01)
                continue
            if scan_region != active_region:
                tracker = MarkerTracker()
                active_region = dict(scan_region)
                journal.record("scan_region", region=scan_region, baseline_existing=True)
            frame_cfg = dict(cfg, region=scan_region)
            img = np.asarray(sct.grab(scan_region))
            markers = find_markers(img, cfg["red_rgb"], cfg["tolerance"], cfg["min_pixels"], "sell")
            markers += find_markers(img, cfg["green_rgb"], cfg["tolerance"], cfg["min_pixels"], "buy")
            if chart_bounds is not None:
                left, top, right, bottom = chart_bounds
                markers = [marker for marker in markers
                           if left <= scan_region["left"] + marker.x < right
                           and top <= scan_region["top"] + marker.y < bottom]
            now = time.monotonic()
            fresh = tracker.update(markers, now)
            remaining = 0 if last_trade is None else max(0, cfg["cooldown_sec"] - (now - last_trade))
            if cfg["scan_bound_chart"]:
                selected, messages = choose_signal(fresh, markers, remaining,
                                                   live_left=max(0, scan_region["width"] - 120))
            else:
                selected, messages = choose_signal(fresh, markers, remaining)
            account_view = guard.monitor.latest
            account_allowed = guard.observe(account_view)
            if guard.reason != last_guard_reason:
                journal.record("position_guard", reason=guard.reason, phase=guard.phase,
                               position=account_view.position,
                               quantity=account_view.quantity,
                               working_orders=account_view.working_orders,
                               read_detail=account_view.reason,
                               entry_surface_clear=account_view.entry_surface_clear,
                               surface_detail=account_view.surface_detail,
                               positions_bounds=account_view.positions_bounds,
                               realized_pnl=account_view.realized_pnl,
                               release_evidence=guard.release_evidence,
                               entry_observed_at=account_view.entry_observed_at,
                               activity_observed_at=guard.activity_at,
                               entry_allowed=account_allowed)
                last_guard_reason = guard.reason
            if selected is not None and not account_allowed:
                account_allowed = guard.refresh_for_signal(account_view)
                journal.record("signal_guard_recheck", entry_allowed=account_allowed, reason=guard.reason)
            if selected is not None and not account_allowed:
                messages.append(guard.reason + "; new signal discarded")
                selected = None
            # Do the action before console output: slow terminal rendering must
            # not hold up an otherwise ready click. enter() logs after clicking.
            if selected is not None:
                seen_at = tracker.first_seen_at(selected)
                journal.record("signal_selected", side=selected.side,
                               screen_x=scan_region["left"] + selected.x,
                               screen_y=scan_region["top"] + selected.y,
                               first_seen_monotonic=seen_at, selected_monotonic=now)
                if not dry:
                    journal.record("click_attempt", side=selected.side, use_ticket=cfg["use_ticket"])
                    try:
                        success = enter(cfg, selected.side, signal_seen_at=seen_at, guard=guard)
                    except BaseException as error:
                        journal.record("click_interrupted", side=selected.side, error_type=type(error).__name__,
                                       execution_verified=False)
                        raise
                    journal.record("click_result", side=selected.side, automation_completed=bool(success),
                                   execution_verified=False)
                    if success is None:
                        # The account changed between selection and the fresh
                        # pre-click check. Do not queue this signal for later.
                        continue
                    if not success:
                        log("Stopped after failed order attempt. Inspect/close the ticket before restarting.")
                        break
                else:
                    journal.record("test_signal", side=selected.side)
                last_trade = time.monotonic()
                log(f"{selected.side.upper()} signal detected at screen x={scan_region['left'] + selected.x:.1f}, "
                    f"y={scan_region['top'] + selected.y:.1f}" + (" (TEST - no click)" if dry else ""))
            for message in tracker.messages + messages:
                journal.record("signal_skipped", reason=message)
                log(message)
            if dry and (test_observation_slot == 0 or fresh):
                try:
                    evidence = save_test_observation(img, frame_cfg, markers, selected,
                                                     tracker.messages + messages, test_observation_slot)
                    label = "startup baseline" if test_observation_slot == 0 else "candidate evidence"
                    log(f"TEST {label}: {evidence}")
                    journal.record("test_evidence", path=str(evidence))
                except OSError as error:
                    log(f"TEST evidence could not be saved: {error}")
                test_observation_slot += 1
            if next_status is None or now >= next_status:
                journal.record("scan_status", buy_count=sum(marker.side == "buy" for marker in markers),
                               sell_count=sum(marker.side == "sell" for marker in markers),
                               cooldown_remaining=remaining, position_guard=guard.reason,
                               reservation_phase=guard.phase)
                log(f"SCAN: visible BUY triangles={sum(marker.side == 'buy' for marker in markers)}, "
                    f"SELL triangles={sum(marker.side == 'sell' for marker in markers)}; "
                    "STATUS ONLY - this is not a new signal; " + account_status(account_view) + "; " + guard.reason)
                next_status = now + 5
            sleep_for = poll_interval - (time.monotonic() - frame_started)
            if sleep_for > 0:
                time.sleep(sleep_for)
    print("Stopped.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=["calibrate", "calibrate-region", "calibrate-buttons",
                        "calibrate-colors", "calibrate-ticket", "check-ticket", "check-position",
                        "reset-position-lock", "run", "test"])
    parser.add_argument("--strategy", default=None,
                        help="User-declared chart version, e.g. 5sec-regime or 5sec-reversed-regime")
    parser.add_argument("--test-seconds", type=float, default=None,
                        help="Stop a read-only TEST automatically after this many seconds")
    parser.add_argument("--reset-position-lock", action="store_true",
                        help="Verify flat/no-orders/quantity 1 and clear the reservation before RUN; simulate in TEST")
    args = parser.parse_args(argv)
    if args.reset_position_lock and args.command not in ("run", "test"):
        parser.error("--reset-position-lock is supported only with run or test")
    if args.test_seconds is not None and (args.command != "test" or not 0 < args.test_seconds <= 120):
        parser.error("--test-seconds requires TEST and a duration in (0, 120]")
    if args.strategy is not None:
        import re
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", args.strategy):
            parser.error("--strategy must be 1-80 ASCII letters, digits, dots, underscores or hyphens")
        if args.command not in ("run", "test"):
            parser.error("--strategy is supported only with run or test")
    cmd = args.command
    run_options = {"strategy_tag": args.strategy}
    if args.reset_position_lock:
        run_options["reset_on_start"] = True
    def launch(**kwargs):
        try:
            run(**kwargs)
        except StartupResetError as error:
            parser.exit(2, f"STARTUP RESET REFUSED: {error}. No order clicks; run not started.\n")
    if cmd == "calibrate":
        calibrate()
    elif cmd == "calibrate-region":
        calibrate_region()
    elif cmd == "calibrate-buttons":
        calibrate_buttons()
    elif cmd == "calibrate-colors":
        calibrate_colors()
    elif cmd == "calibrate-ticket":
        calibrate_ticket()
    elif cmd == "check-ticket":
        check_ticket()
    elif cmd == "check-position":
        check_position()
    elif cmd == "reset-position-lock":
        check_position(reset=True)
    elif cmd == "run":
        launch(**run_options)
    elif cmd == "test":
        if args.test_seconds is not None:
            launch(dry=True, test_seconds=args.test_seconds, **run_options)
        else:
            launch(dry=True, **run_options)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C. No further order clicks.")
