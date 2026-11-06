"""Read-only TradingView account guard; no broker API and no UI input methods.

Entry permission requires positive evidence of FLAT, an empty working-order
badge, the bound account, the bound page, and quantity exactly one. A durable
reservation is written BEFORE the click. A reservation cannot expire by time.
"""
import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlsplit
from entry_surface import DesktopQuoteReader

ROOT = Path(__file__).resolve().parent
EMPTY = "Nothing to show yet. Your account activity will appear here."
EMPTY_TEXTS = {EMPTY, "Make a trade to get started Buy or sell — choose your pill."}
EMPTY_TEXTS.add("Make a trade to get started Buy or sell — choose your pill. Sell or Buy")


@dataclass(frozen=True)
class AccountView:
    observed_at: float
    account: str = ""
    broker: str = ""
    position: str = "unknown"
    quantity: int | None = None
    working_orders: int | None = None
    foreground: bool = False
    page_ok: bool = False
    reason: str = "Account view unavailable"
    buy_point: tuple | None = None
    sell_point: tuple | None = None
    entry_surface_clear: bool | None = None
    chart_bounds: tuple | None = None
    surface_detail: str = ""
    positions_bounds: tuple | None = None
    realized_pnl: str | None = None
    entry_observed_at: float | None = None


def money_value(value):
    """Canonical exact account amount. Unreadable/hidden values stay unknown."""
    if not isinstance(value, str):
        return None
    value = value.strip().replace("\u2212", "-")
    if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", value):
        return None
    try:
        amount = Decimal(value.replace(",", ""))
        return "0" if amount == 0 else format(amount.normalize(), "f")
    except InvalidOperation:
        return None


def entry_surface_available(view):
    # Native observations supply an explicit visibility/hit-test result.
    # The fallback keeps legacy fixtures compatible, not native UI failures.
    return view.foreground if view.entry_surface_clear is None else view.entry_surface_clear


def account_status(view):
    """Account evidence is separate from permission to click a screen point."""
    orders = "?" if view.working_orders is None else str(view.working_orders)
    quantity = "?" if view.quantity is None else str(view.quantity)
    return f"ACCOUNT: position={view.position.upper()}, orders={orders}, entry quantity={quantity}"


def intersect_scan(region, bounds):
    left = max(region["left"], bounds[0])
    top = max(region["top"], bounds[1])
    right = min(region["left"] + region["width"], bounds[2])
    bottom = min(region["top"] + region["height"], bounds[3])
    return (left, top, right, bottom) if right > left and bottom > top else None


def region_probes(bounds):
    left, top, right, bottom = bounds
    return [(x, y) for x in (left, (left + right - 1) // 2, right - 1)
            for y in (top, (top + bottom - 1) // 2, bottom - 1)]


def signal_scan_region(cfg, bounds):
    if not cfg.get("scan_bound_chart", False):
        return cfg["region"]
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    if right - left < 180 or bottom - top < 120:
        return None
    return dict(left=left + 8, top=top + 4,
                width=right-left-98, height=bottom-top-30)


def approved_accounts(cfg):
    binding = cfg.get("position_guard", {})
    pairs = {(item.get("account"), item.get("broker")) for item in binding.get("accounts", [])}
    if binding.get("account") and binding.get("broker"):
        pairs.add((binding["account"], binding["broker"]))
    return {(account, broker) for account, broker in pairs if account and broker}


def parse_view(data, observed_at, cfg):
    """Strict parsing, shared by native observations and offline fixtures."""
    account, broker = data.get("account", ""), data.get("broker", "")
    position = "unknown"
    reason = "Positions tab must be selected and readable"
    count = data.get("position_count")
    # A positive Positions badge is activity evidence even on Orders/summary.
    # Absence of a badge is never sufficient to assert FLAT.
    if ((not data.get("positions_selected") or not data.get("table_readable"))
            and isinstance(count, int) and count > 0):
        position, reason = "open", "Positive Positions badge (position rows unavailable)"
    elif data.get("positions_selected") and data.get("table_readable"):
        count = data.get("position_count")
        empty = data.get("empty_text") in EMPTY_TEXTS
        rows = data.get("position_rows", [])
        if empty and count in (None, 0) and not rows:
            position, reason = "flat", "No open positions visible"
        elif not empty and (isinstance(count, int) and count > 0 or rows):
            position, reason = "open", "Open account position"
        else:
            reason = "Conflicting or incomplete position evidence"
    if (account, broker) not in approved_accounts(cfg):
        position, reason = "unknown", "Account or broker differs from bound configuration"
    tab_stable = data.get("tab_stable", True)
    if not tab_stable:
        # Positive activity can only block/acknowledge, never grant entry.
        # A tab transition must not erase an OPEN captured for this account.
        if position != "open":
            position = "unknown"
        reason += "; account tab changed during read"
    return AccountView(observed_at, account, broker, position,
                       data.get("quantity"), data.get("working_orders"),
                       bool(data.get("foreground")), bool(data.get("page_ok")), reason,
                       data.get("buy_point"), data.get("sell_point"),
                       data.get("entry_surface_clear") if tab_stable else False, data.get("chart_bounds"),
                       data.get("surface_detail", ""), data.get("positions_bounds"),
                       money_value(data.get("realized_pnl")), data.get("entry_observed_at"))


def badge_count(name, child_names, prefix, absent=None):
    """Chrome may expose a badge in the tab name OR a separate text node."""
    match = re.fullmatch(re.escape(prefix) + r"(?:\s+(\d+))?", name)
    if not match:
        return None
    values = {int(value) for value in child_names if re.fullmatch(r"\d+", value)}
    if match[1] is not None:
        values.add(int(match[1]))
    return next(iter(values)) if len(values) == 1 else absent if not values else None


def is_position_row(name):
    """A real position row begins with symbol, side and nonzero quantity."""
    return bool(re.match(r"^[A-Z0-9_:!./-]+\s+(?:Long|Short|Buy|Sell)\s+[1-9]\d*(?:\.\d+)?\s", name))


class WindowsAccountReader:
    """Microsoft UI Automation reads only. Never selects, invokes, or clicks.

    Chrome must expose accessibility and keep the bound Positions pane open.
    Missing controls are UNKNOWN, never interpreted as a closed position.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.binding = cfg.get("position_guard", {})
        self.auto = None
        self.window = None
        self.manager = None
        self.quantity_detail = "not read"
        self.desktop_quotes = None
        self.cached_entry = {}

    def _walk(self, root, limit=1600, depth=35):
        deadline = time.monotonic() + 1.2
        stack = [(root, 0)]
        total = 0
        while stack:
            node, level = stack.pop()
            total += 1
            if total > limit or time.monotonic() > deadline:
                raise RuntimeError("Accessibility traversal incomplete")
            yield node
            if level < depth:
                stack.extend((child, level + 1) for child in reversed(node.GetChildren()))

    @staticmethod
    def _visible(node):
        rect = node.BoundingRectangle
        return not node.IsOffscreen and rect.right > rect.left and rect.bottom > rect.top

    @staticmethod
    def _has_layout(node):
        rect = node.BoundingRectangle
        return rect.right > rect.left and rect.bottom > rect.top

    def _table_data(self, table):
        # A selected Positions grid can extend BELOW the viewport. Windows
        # accessibility still exposes its current rows. IsOffscreen is about
        # clipping, not whether the account is flat. CSS-hidden, zero-layout
        # grids and nodes are excluded; mixed evidence remains UNKNOWN.
        result = dict(table_readable=False, empty_text="", position_rows=[])
        if not self._has_layout(table):
            return result
        result["table_readable"] = True
        for node in self._walk(table, limit=250, depth=12):
            if not self._has_layout(node):
                continue
            name = " ".join(node.Name.split())
            if name in EMPTY_TEXTS:
                result["empty_text"] = name
            elif node.ControlTypeName == "DataItemControl" and is_position_row(name):
                result["position_rows"].append(name)
        return result

    def _realized_value(self, label):
        # Only the amount immediately following the exact Realized PnL label.
        # Unrealized PnL/equity changes cannot acknowledge a closed trade.
        siblings = label.GetParentControl().GetChildren()
        for index, node in enumerate(siblings[:-1]):
            if node.Name != "Realized PnL":
                continue
            values = {money_value(child.Name) for child in self._walk(siblings[index + 1], limit=40, depth=5)
                      if self._has_layout(child)}
            values.discard(None)
            return next(iter(values)) if len(values) == 1 else None
        return None

    def _bind(self):
        candidates = [node for node in self.auto.GetRootControl().GetChildren()
                      if self.binding["window_title"] in node.Name
                      and node.ClassName == "Chrome_WidgetWin_1"]
        if len(candidates) != 1:
            raise RuntimeError("Expected exactly one bound Chrome window")
        self.window = candidates[0]
        self.manager = None
        self.address = None
        self.document = None
        self.broker_control = None
        self.chart = None
        self.cached_entry = {}
        brokers = {broker for account, broker in approved_accounts(self.cfg)}
        for node in self._walk(self.window):
            name = node.Name
            if name == "Account manager" and self._visible(node):
                self.manager = node
            if node.ControlTypeName == "EditControl" and name in ("Address and search bar", "Address bar"):
                self.address = node
            if node.ControlTypeName == "DocumentControl":
                self.document = node
            if name == self.binding.get("chart_name", "Chart #4"):
                self.chart = node
            if name in brokers and self._visible(node):
                self.broker_control = node
            if self.manager and self.chart and (self.address or self.document) and self.broker_control:
                break
        # Account discovery is independent of the entry chart/header. Search
        # the desktop's bound window, including windows spanning monitors.
        if not self.manager or not (self.address or self.document) or not self.broker_control:
            raise RuntimeError(f"Accessibility binding unavailable: account_pane={bool(self.manager)}, address={bool(self.address)}, document={bool(self.document)}, broker={bool(self.broker_control)}")

    def _quote_widget(self):
        # Locate the actual quantity field BETWEEN the two quote controls of
        # the configured chart. No OCR and no assumption that an old screen
        # coordinate still points to the quantity selector or an order button.
        widgets = []
        self.quote_controls = []
        descriptions = set(self.binding.get("chart_descriptions", []))
        if self.binding.get("chart_description"):
            descriptions.add(self.binding["chart_description"])
        chart_description_ok = not descriptions
        actual_description = "unavailable"
        for node in self._walk(self.chart, limit=500, depth=15):
            if node.Name.startswith("Chart for "):
                actual_description = node.Name
            if node.Name in descriptions:
                chart_description_ok = True
            if node.ControlTypeName != "TextControl" or not re.fullmatch(r"[1-9]\d*", node.Name.strip()):
                continue
            group = node.GetParentControl()
            rect = group.BoundingRectangle
            if not self._visible(group) or not (18 <= rect.right - rect.left <= 60 and 18 <= rect.bottom - rect.top <= 45):
                continue
            controls = group.GetChildren()
            if len(controls) != 1 or controls[0].Name != node.Name:
                continue
            left, right = [], []
            for sibling in group.GetParentControl().GetChildren():
                area = sibling.BoundingRectangle
                if not self._visible(sibling) or not (50 <= area.right - area.left <= 250 and 25 <= area.bottom - area.top <= 55):
                    continue
                if abs((area.top + area.bottom - rect.top - rect.bottom) / 2) > 12:
                    continue
                if area.right <= rect.left:
                    left.append(sibling)
                if area.left >= rect.right:
                    right.append(sibling)
            if len(left) == len(right) == 1:
                def midpoint(control):
                    bounds = control.BoundingRectangle
                    return (round((bounds.left + bounds.right) / 2), round((bounds.top + bounds.bottom) / 2))
                def quote_point(control):
                    bounds = control.BoundingRectangle
                    return (bounds.left + 6, round((bounds.top + bounds.bottom) / 2))
                widgets.append((int(node.Name), quote_point(right[0]), quote_point(left[0]), right[0], left[0], group))
        if len(widgets) != 1 or not chart_description_ok:
            self.quantity_detail = (f"Unexpected chart: {actual_description}" if not chart_description_ok else
                                    "Expected one quantity field between the chart's two quote buttons")
            return None, None, None
        self.quantity_detail = "Quantity and quote coordinates read from bound chart header"
        self.quote_controls = [(widgets[0][1], widgets[0][3]), (widgets[0][2], widgets[0][4]),
                               (midpoint(widgets[0][5]), widgets[0][5])]
        return widgets[0][:3]

    def _quote_surface(self, control, point, side):
        if not self._visible(control) or not control.IsEnabled:
            return False, "Quote control hidden or disabled"
        area = control.BoundingRectangle
        if self.desktop_quotes is None:
            self.desktop_quotes = DesktopQuoteReader()
        return self.desktop_quotes.check((area.left, area.top, area.right, area.bottom), side, point)

    def close(self):
        if self.desktop_quotes is not None:
            self.desktop_quotes.close()
            self.desktop_quotes = None

    def _surface_clear(self):
        # A window spanning monitors can be visible while another window owns
        # keyboard focus. Verify the actual desktop points, not keyboard focus.
        bounds = self.chart.BoundingRectangle
        chart_bounds = (bounds.left, bounds.top, bounds.right, bounds.bottom)
        region = signal_scan_region(self.cfg, chart_bounds)
        scan = intersect_scan(region, chart_bounds) if region else None
        if not scan or len(self.quote_controls) != 3:
            return False, chart_bounds, "Chart scan area or quote controls unavailable"
        expected_handle = self.window.NativeWindowHandle

        def owner_at(point):
            handle = self.auto.WindowFromPoint(*point)
            return self.auto.GetAncestor(handle, 2) if handle else 0

        for point, control in self.quote_controls:
            if owner_at(point) != expected_handle:
                return False, chart_bounds, f"Another window covers order/quantity point {point}"
            if not self._visible(control) or not control.IsEnabled:
                return False, chart_bounds, "Order/quantity control hidden or disabled"
        for side, (point, control) in zip(("buy", "sell"), self.quote_controls[:2]):
            clear, detail = self._quote_surface(control, point, side)
            if not clear:
                return False, chart_bounds, f"Quote appearance unverified: {detail}"
            # Recheck ownership after capture; a newly covering window fails.
            if owner_at(point) != expected_handle:
                return False, chart_bounds, f"Another window covered {side.upper()} during verification"
        for point in region_probes(scan):
            if owner_at(point) != expected_handle:
                return False, chart_bounds, f"Another window covers chart scan point {point}"
        return True, chart_bounds, "Quote cards verified from desktop pixels; scan area uncovered"

    def _entry_data(self):
        """Entry controls can fail without erasing a readable Positions pane."""
        result = dict(quantity=None, buy_point=None, sell_point=None,
                      entry_surface_clear=False, chart_bounds=None,
                      surface_detail="Entry chart unavailable")
        try:
            if self.chart is None:
                for node in self._walk(self.window):
                    if node.Name == self.binding.get("chart_name", "Chart #4"):
                        self.chart = node
                        break
            if self.chart is not None:
                quantity, buy_point, sell_point = self._quote_widget()
                # Preserve the readable quantity even if hit-testing fails.
                result.update(quantity=quantity, buy_point=buy_point, sell_point=sell_point)
                clear, bounds, detail = self._surface_clear()
                result.update(entry_surface_clear=clear, chart_bounds=bounds, surface_detail=detail)
        except Exception as error:
            result["surface_detail"] = f"Entry controls unverified: {type(error).__name__}"
            self.chart = None
        return result

    def read(self, include_entry=True):
        started = time.monotonic()
        try:
            if self.auto is None:
                import uiautomation as auto
                self.auto = auto
            if not self.window:
                self._bind()
            # Discovery can be slow on a cold UIA connection. Date the actual
            # account traversal, not the preceding desktop-window search.
            started = time.monotonic()
            # A fresh address-bar value prevents accepting another page in a
            # tab with a similar title. No browser internals or HTTP requests.
            url = (self.address.GetValuePattern().Value if self.address else
                   self.document.GetLegacyIAccessiblePattern().Value).strip()
            def url_identity(value):
                parsed = urlsplit(value if "://" in value else "https://" + value)
                return (parsed.hostname or "").removeprefix("www."), parsed.path.rstrip("/"), parsed.query
            page_ok = url_identity(url) == url_identity(self.binding["page_url"])
            foreground = self.auto.GetForegroundControl().GetTopLevelControl()
            front_ok = bool(foreground and foreground.NativeWindowHandle == self.window.NativeWindowHandle)
            broker_name = self.broker_control.Name
            expected_accounts = {account for account, broker in approved_accounts(self.cfg) if broker == broker_name}
            data = dict(account="", broker="", foreground=front_ok,
                        page_ok=page_ok,
                        positions_selected=False, table_readable=False,
                        position_count=None, working_orders=None,
                        empty_text="", position_rows=[])
            if self._visible(self.broker_control):
                data["broker"] = broker_name
            else:
                self.window = None
            tables = []
            realized_values = set()
            connected_label = ""
            account_control = positions_control = None
            for node in self._walk(self.manager, limit=700, depth=18):
                name, kind = node.Name.strip(), node.ControlTypeName
                if kind in ("TableControl", "DataGridControl") and self._has_layout(node):
                    tables.append(node)
                if not self._visible(node):
                    continue
                if kind == "TextControl" and name == "Realized PnL":
                    try:
                        amount = self._realized_value(node)
                    except Exception:
                        # Optional closure evidence must not break position
                        # detection when a header rerenders or hides values.
                        amount = None
                    if amount is not None:
                        realized_values.add(amount)
                if kind == "ButtonControl":
                    if re.fullmatch(r"\S+\s+[-+\u2212\d,.]+\s+[A-Z]{3}", name):
                        connected_label = name
                    for account in expected_accounts:
                        if name == account or name.startswith(account + " "):
                            data["account"] = account
                            account_control = node
                if kind == "TabItemControl":
                    if re.fullmatch(r"Positions(?:\s+\d+)?", name):
                        data["positions_selected"] = node.GetSelectionItemPattern().IsSelected
                        positions_control = node
                        children = [child.Name.strip() for child in self._walk(node, limit=30, depth=4)]
                        data["position_count"] = badge_count(name, children, "Positions")
                    if re.fullmatch(r"Orders(?:\s+\d+)?", name):
                        children = [child.Name.strip() for child in self._walk(node, limit=30, depth=4)]
                        data["working_orders"] = badge_count(name, children, "Orders", absent=0)
            if len(tables) == 1:
                data.update(self._table_data(tables[0]))
                bounds = tables[0].BoundingRectangle
                data["positions_bounds"] = (bounds.left, bounds.top, bounds.right, bounds.bottom)
            if len(realized_values) == 1:
                data["realized_pnl"] = next(iter(realized_values))
            # Read position/account evidence BEFORE consulting quote buttons.
            if include_entry:
                entry_started = time.monotonic()
                self.cached_entry = self._entry_data()
                self.cached_entry["entry_observed_at"] = entry_started
                # A rebuilt header must be located anew after a failed read.
                if self.cached_entry.get("quantity") is None:
                    self.chart = None
            data.update(self.cached_entry)
            data["tab_stable"] = bool(positions_control and
                                      positions_control.GetSelectionItemPattern().IsSelected == data["positions_selected"])
            # Account switches and tab changes during a traversal invalidate
            # the entire observation rather than mixing two account views.
            if (not account_control or not positions_control or not self._visible(self.manager)
                    or self.broker_control.Name != data["broker"]
                    or not self._visible(self.broker_control)
                    or not (account_control.Name == data["account"] or account_control.Name.startswith(data["account"] + " "))):
                self.window = None
                detail = (f"Connected account is not approved: {connected_label}" if not account_control and connected_label else
                          "Account/tab changed during read")
                return replace(parse_view(data, started, self.cfg), position="unknown",
                               entry_surface_clear=False, reason=detail)
            if not data["tab_stable"]:
                self.window = None
            result = parse_view(data, started, self.cfg)
            if result.position == "unknown" or result.quantity is None:
                result = replace(result, reason=result.reason + "; " + self.quantity_detail +
                                 f"; selected={data['positions_selected']}, tables={len(tables)}, broker={bool(data['broker'])}, account={bool(data['account'])}")
            return result
        except Exception as error:
            self.window = self.manager = None
            return AccountView(started, reason=f"Account read failed: {type(error).__name__}: {error}")


class AccountMonitor:
    """UIA lives in its own COM-initialized thread; screenshots keep scanning."""
    def __init__(self, cfg, reader_factory=WindowsAccountReader):
        self.cfg, self.reader_factory = cfg, reader_factory
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.requested = threading.Event()
        self.latest = AccountView(0)
        self.sequence = 0
        self.entry_sequence = 0
        self.request_generation = 0
        self.entry_generation = 0
        self.entry_latest = AccountView(0)
        self.activities = deque(maxlen=128)
        self.last_activity_signature = None
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._worker, daemon=True, name="account-read-only")
        self.thread.start()
        return self

    def _worker(self):
        try:
            import uiautomation as auto
            with auto.UIAutomationInitializerInThread():
                reader = self.reader_factory(self.cfg)
                try:
                    next_entry_check = 0
                    while not self.stopped.is_set():
                        with self.condition:
                            generation = self.request_generation
                            full = generation > self.entry_generation or time.monotonic() >= next_entry_check
                            if full:
                                self.requested.clear()
                        view = reader.read(include_entry=full)
                        self._publish(view, generation if full else None)
                        if full:
                            next_entry_check = time.monotonic() + .25
                        self.requested.wait(.02)
                finally:
                    reader.close()
        except Exception as error:
            with self.condition:
                self.latest = AccountView(time.monotonic(), reason=f"Account monitor failed: {error}")
                self.sequence += 1
                self.condition.notify_all()

    def _publish(self, view, entry_generation=None):
        with self.condition:
            self.latest = view
            self.sequence += 1
            active = view.position == "open" or (view.working_orders is not None and view.working_orders > 0)
            signature = (view.account, view.broker, view.position, view.working_orders) if active else None
            if active and signature != self.last_activity_signature:
                self.activities.append(view)
            self.last_activity_signature = signature
            if entry_generation is not None:
                self.entry_latest = view
                self.entry_generation = entry_generation
                self.entry_sequence += 1
            self.condition.notify_all()

    def drain_activity(self):
        with self.condition:
            result = list(self.activities)
            self.activities.clear()
            return result

    def fresh(self, timeout=2.0):
        with self.condition:
            self.request_generation += 1
            target = self.request_generation
            self.requested.set()
            if not self.condition.wait_for(lambda: self.entry_generation >= target, timeout=timeout):
                return AccountView(time.monotonic(), reason="Fresh entry/account check timed out")
            return self.entry_latest

    def __exit__(self, *args):
        self.stopped.set()
        self.requested.set()
        if self.thread:
            self.thread.join(timeout=0.2)


class BotInstanceLock:
    """OS-held byte lock; even distinct config files cannot start two RUNs."""
    def __init__(self, path=ROOT / "nq_bot_single_instance.lock"):
        self.path = Path(path)

    def __enter__(self):
        import msvcrt
        self.stream = self.path.open("a+b")
        try:
            if os.fstat(self.stream.fileno()).st_size == 0:
                self.stream.write(b"0")
                self.stream.flush()
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.stream.close()
            raise RuntimeError("Another guarded RUN is already running")
        return self

    def __exit__(self, *args):
        import msvcrt
        self.stream.seek(0)
        msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        self.stream.close()


class PositionGate:
    """Activity/closure acknowledgement, then two fresh FLAT/no-order reads.

    An unacknowledged reservation never expires by time. Closed fast trades
    may be acknowledged by a changed realized PnL from the pre-click baseline.
    """
    def __init__(self, cfg, monitor, state_path=ROOT / "nq_bot_position_state.json", dry=False, clock=time.monotonic):
        self.cfg, self.monitor, self.clock = cfg, monitor, clock
        self.path = Path(state_path) if state_path else None
        self.dry = dry
        self.account = cfg.get("position_guard", {}).get("account")
        self.broker = cfg.get("position_guard", {}).get("broker")
        self.phase = "flat"
        self.reserved_at = -1
        self.claimed_view = None
        self.last_seen = -1
        self.flat_reads = 0
        self.last_flat_at = -1
        self.reservation_realized_pnl = None
        self.release_evidence = ""
        self.activity_at = -1
        self.reason = "Waiting for account read"
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if data.get("phase") not in ("flat", "pending", "open", "working"):
                    raise ValueError("Invalid guard phase")
                if data.get("phase") != "flat" and (data.get("account"), data.get("broker")) not in approved_accounts(cfg):
                    raise ValueError("Unresolved reservation for a different account")
                self.phase = data["phase"]
                if self.phase != "flat":
                    self.account, self.broker = data["account"], data["broker"]
                if self.phase == "pending":
                    self.reservation_realized_pnl = money_value(data.get("reservation_realized_pnl"))
            except Exception as error:
                raise RuntimeError(f"Position lock cannot be trusted: {error}") from error

    def _save(self, phase):
        if not self.dry and self.path:
            data = dict(phase=phase, account=self.account, broker=self.broker,
                        updated_epoch=time.time(), max_contracts=1)
            if phase == "pending":
                data["reservation_realized_pnl"] = self.reservation_realized_pnl
            temporary = self.path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(data, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        self.phase = phase
        if phase == "flat":
            self.reservation_realized_pnl = None

    def observe(self, view):
        if (self.account is None and (view.account, view.broker) in approved_accounts(self.cfg)
                and view.page_ok and 0 <= self.clock() - view.observed_at <= 0.75):
            # Bind ONCE per run. Switching accounts later blocks until restart.
            self.account, self.broker = view.account, view.broker
        # The capture loop may be slower than a very short open/close cycle.
        # Consume latched positive activity before examining the latest FLAT.
        drain = getattr(type(self.monitor), "drain_activity", None)
        if drain is not None:
            for activity in self.monitor.drain_activity():
                if (activity.account == self.account and activity.broker == self.broker
                        and activity.page_ok and 0 <= activity.observed_at <= self.clock()
                        and (self.phase != "pending" or activity.observed_at > self.reserved_at)):
                    if activity.position == "open":
                        if self.phase != "open":
                            self._save("open")
                    elif activity.working_orders is not None and activity.working_orders > 0:
                        if self.phase in ("pending", "flat"):
                            self._save("working")
                    else:
                        continue
                    self.activity_at = max(self.activity_at, activity.observed_at)
                    self.flat_reads = 0
                    self.last_flat_at = -1
        identity_ok = (view.account == self.account and view.broker == self.broker and view.page_ok)
        age = self.clock() - view.observed_at
        # Positive activity only locks the account. Do not discard an OPEN
        # seen after reservation because the slower quote hit-test took time.
        if identity_ok and age >= 0 and view.position == "open":
            self.flat_reads = 0
            if self.phase == "pending" and view.observed_at <= self.reserved_at:
                self.reason = "BLOCKED: old position observation predates reservation"
                return False
            if self.phase != "open":
                self._save("open")
            self.activity_at = max(self.activity_at, view.observed_at)
            self.reason = "BLOCKED: open position (BUY and SELL ignored)"
            return False
        if not identity_ok or not 0 <= age <= 0.75:
            self.flat_reads = 0
            self.reason = "BLOCKED: account/page unavailable or stale"
            return False
        if view.working_orders is not None and view.working_orders > 0:
            self.flat_reads = 0
            if self.phase == "pending" and view.observed_at <= self.reserved_at:
                self.reason = "BLOCKED: old order observation predates reservation"
                return False
            if self.phase == "pending":
                self._save("working")
            self.activity_at = max(self.activity_at, view.observed_at)
            self.reason = "BLOCKED: working orders (BUY and SELL ignored)"
            return False
        if view.position != "flat" or view.working_orders != 0:
            self.flat_reads = 0
            self.reason = "BLOCKED: unreadable position or working orders"
            return False
        if view.observed_at <= self.activity_at:
            self.flat_reads = 0
            self.reason = "BLOCKED: flat observation predates recorded position/order activity"
            return False
        current_pnl = money_value(view.realized_pnl)
        pnl_changed = (self.phase == "pending" and self.reservation_realized_pnl is not None
                       and current_pnl is not None
                       and current_pnl != self.reservation_realized_pnl
                       and view.observed_at > self.reserved_at)
        if self.phase == "pending" and not pnl_changed:
            self.flat_reads = 0
            self.reason = "BLOCKED: entry reserved; waiting for position/order activity or verified closure"
            return False
        if view.observed_at > self.last_seen:
            self.last_seen = view.observed_at
            if self.last_flat_at < 0 or view.observed_at - self.last_flat_at >= 0.10:
                self.flat_reads += 1
                self.last_flat_at = view.observed_at
        if self.phase in ("open", "working", "pending") and self.flat_reads >= 2:
            self.release_evidence = ("realized_pnl_changed_and_two_flat_reads" if pnl_changed else
                                     "working_orders_cleared_and_two_flat_reads" if self.phase == "working" else
                                     "position_closed_and_two_flat_reads")
            self._save("flat")
        # OPEN -> FLAT is determined by the account pane, even when entry
        # controls are covered or missing. Clicking still requires both checks.
        if self.phase == "pending":
            self.reason = "BLOCKED: closure detected; waiting for two fresh flat observations"
            return False
        if view.quantity != 1:
            self.reason = "BLOCKED: entry quantity is not exactly 1"
            return False
        if not self._entry_is_fresh(view):
            self.reason = "BLOCKED: entry controls need a fresh check"
            return False
        if not entry_surface_available(view):
            self.reason = "BLOCKED: " + (view.surface_detail or "TradingView controls or scan area are covered/unverified")
            return False
        allowed = self.phase == "flat" and self.flat_reads >= 2
        self.reason = ("READY: flat, no working orders, quantity 1" if allowed else
                       "BLOCKED: entry reserved; waiting to observe open then closed" if self.phase == "pending" else
                       "BLOCKED: waiting for two fresh flat observations")
        return allowed

    def _entry_is_fresh(self, view):
        return view.entry_observed_at is None or 0 <= self.clock() - view.entry_observed_at <= .75

    def _visual_retry_allowed(self, view):
        # Only an otherwise verified, flat account may refresh a negative
        # visual observation. OPEN, pending, orders and unknown never queue.
        return (self.phase == "flat" and view.position == "flat"
                and view.working_orders == 0 and view.quantity == 1
                and view.account == self.account and view.broker == self.broker
                and view.page_ok and 0 <= self.clock() - view.observed_at <= 0.75
                and (not entry_surface_available(view) or not self._entry_is_fresh(view)))

    def _fresh_entry_check(self):
        for _ in range(2):
            view = self.monitor.fresh()
            if self.observe(view):
                return view
            if not self._visual_retry_allowed(view):
                break
        return None

    def refresh_for_signal(self, view):
        """Resolve a new signal against fresh UI, without reserving/clicking."""
        if not self._visual_retry_allowed(view):
            return False
        return self._fresh_entry_check() is not None

    def claim(self):
        # Re-read immediately before claiming; do not click using a cached UI.
        view = self._fresh_entry_check()
        if view is None:
            return False
        if self.cfg.get("use_ticket", False):
            self.reason = "BLOCKED: ticket quantity is not validated; use calibrated one-click mode"
            return False
        if not view.buy_point or not view.sell_point:
            self.reason = "BLOCKED: chart quote buttons not verified"
            return False
        self.reservation_realized_pnl = money_value(view.realized_pnl)
        self.release_evidence = ""
        self._save("pending")  # Durable before any order-button click.
        self.reserved_at = self.clock()
        self.claimed_view = view
        self.flat_reads = 0
        self.reason = "BLOCKED: entry reserved before click"
        return True

    def entry_config(self, cfg):
        if self.phase != "pending" or self.claimed_view is None:
            raise RuntimeError("No claimed entry coordinates")
        return dict(cfg, buy_btn=self.claimed_view.buy_point, sell_btn=self.claimed_view.sell_point)

    def still_flat(self):
        """Ticket confirmation recheck; a reservation alone is not permission."""
        view = self.monitor.fresh()
        return (self.phase == "pending" and view.account == self.account
                and view.broker == self.broker and view.position == "flat"
                and view.working_orders == 0 and view.quantity == 1
                and entry_surface_available(view) and view.page_ok
                and self._entry_is_fresh(view)
                and 0 <= self.clock() - view.observed_at <= 0.75)

    def reset(self):
        """Explicit read-only recovery: no click permission is granted by reset.

        Window focus/coverage matters for clicking, not reading the bound
        account. Recovery still requires 3 independent fresh FLAT snapshots.
        A subsequent entry must independently pass the full visibility guard.
        """
        self.flat_reads = 0
        self.last_flat_at = -1
        independent = 0
        previous = -1
        for _ in range(10):
            view = self.monitor.fresh()
            # A clear/brand-new lock binds to the approved account connected
            # now. An unresolved lock stays bound to its original account.
            if (self.account is None and self.broker is None
                    and (view.account, view.broker) in approved_accounts(self.cfg)
                    and view.page_ok and view.position == "flat" and view.working_orders == 0
                    and view.quantity == 1 and self._entry_is_fresh(view)
                    and 0 <= self.clock() - view.observed_at <= .75):
                self.account, self.broker = view.account, view.broker
            # Recovery never overrides a positive OPEN or unknown UI state.
            if (view.position != "flat" or view.working_orders != 0 or view.quantity != 1
                    or view.account != self.account or view.broker != self.broker
                    or not view.page_ok
                    or not self._entry_is_fresh(view)
                    or not 0 <= self.clock() - view.observed_at <= 0.75):
                raise RuntimeError(f"Cannot reset: position={view.position}, orders={view.working_orders}, "
                                   f"quantity={view.quantity}, account={view.account}/{view.broker}, "
                                   f"expected={self.account}/{self.broker}; fresh same-account FLAT not verified; {view.reason}")
            if previous < 0 or view.observed_at - previous >= 0.10:
                independent += 1
                previous = view.observed_at
            if independent >= 3:
                break
        if independent < 3:
            raise RuntimeError("Cannot reset: flat observations not independent")
        self._save("flat")
        self.flat_reads = 0
        self.reason = "Reset verified; waiting for independent entry-surface checks"


def configuration_errors(cfg):
    binding = cfg.get("position_guard", {})
    missing = [key for key in ("window_title", "page_url") if not binding.get(key)]
    if not approved_accounts(cfg):
        missing.append("accounts")
    return ["Missing position guard binding: " + ", ".join(missing)] if missing else []
