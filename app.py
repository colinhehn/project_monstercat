"""Textual UI for generating audio visualizer videos."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Rule,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from visualizers import get_visualizer, list_visualizers


class ConfigForm(VerticalScroll):
    """Dynamic form built from a visualizer CONFIG_SCHEMA."""

    DEFAULT_CSS = """
    ConfigForm {
        height: 1fr;
        padding: 0 1;
    }
    ConfigForm .field-row {
        height: auto;
        margin-bottom: 1;
    }
    ConfigForm .field-label {
        width: 28;
        color: $text-muted;
    }
    ConfigForm Input {
        width: 1fr;
    }
    """

    def __init__(self, schema: list[dict[str, Any]], defaults: dict[str, Any], **kwargs: Any):
        super().__init__(**kwargs)
        self.schema = schema
        self.defaults = defaults

    def compose(self) -> ComposeResult:
        groups: dict[str, list[dict[str, Any]]] = {}
        for field in self.schema:
            groups.setdefault(field["group"], []).append(field)

        with TabbedContent():
            for group, fields in groups.items():
                with TabPane(group, id=f"tab-{group.lower()}"):
                    for field in fields:
                        yield from self._compose_field(field)

    def _compose_field(self, field: dict[str, Any]):
        key = field["key"]
        label = field["label"]
        ftype = field["type"]
        default = self.defaults.get(key, field["default"])
        widget_id = f"cfg-{key}"

        with Horizontal(classes="field-row"):
            yield Label(label, classes="field-label")
            if ftype == "bool":
                yield Switch(value=bool(default), id=widget_id)
            else:
                yield Input(value="" if default is None else str(default), id=widget_id)

    def read_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for field in self.schema:
            key = field["key"]
            ftype = field["type"]
            widget_id = f"cfg-{key}"
            if ftype == "bool":
                config[key] = self.query_one(f"#{widget_id}", Switch).value
                continue

            raw = self.query_one(f"#{widget_id}", Input).value.strip()
            if ftype == "int":
                config[key] = int(raw)
            elif ftype == "float":
                config[key] = float(raw)
            elif ftype == "optional_int":
                config[key] = "" if raw == "" else int(raw)
            elif ftype in ("str", "color"):
                config[key] = raw
            else:
                config[key] = raw
        return config

    def apply_defaults(self, defaults: dict[str, Any] | None = None) -> None:
        values = defaults if defaults is not None else self.defaults
        for field in self.schema:
            key = field["key"]
            ftype = field["type"]
            widget_id = f"cfg-{key}"
            default = values.get(key, field["default"])
            if ftype == "bool":
                self.query_one(f"#{widget_id}", Switch).value = bool(default)
            else:
                self.query_one(f"#{widget_id}", Input).value = (
                    "" if default is None else str(default)
                )


class VisualizerApp(App[None]):
    """Lightweight TUI to pick audio, tune config, and render a visualizer."""

    TITLE = "Audio Visualizer"
    SUB_TITLE = "Generate promo visualizers"
    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        height: 1fr;
        padding: 1 2;
    }
    #sidebar {
        width: 1fr;
        min-width: 40;
        height: 1fr;
    }
    #config-pane {
        width: 2fr;
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    #paths {
        height: auto;
        margin-bottom: 1;
    }
    #paths Input {
        width: 1fr;
        margin-bottom: 1;
    }
    #actions {
        height: auto;
        margin-top: 1;
    }
    #actions Button {
        margin-right: 1;
    }
    #status-row {
        height: auto;
        margin-top: 1;
    }
    #status-label {
        color: $text-muted;
        margin-bottom: 1;
    }
    #log {
        height: 10;
        margin-top: 1;
        border: solid $surface;
    }
    .section-title {
        text-style: bold;
        margin-bottom: 1;
        color: $accent;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("g", "generate", "Generate"),
    ]

    busy: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self._visualizers = list_visualizers()
        self._current_id = self._visualizers[0].id

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Static("Setup", classes="section-title")
                yield Label("Visualizer")
                yield Select(
                    options=[(v.name, v.id) for v in self._visualizers],
                    value=self._current_id,
                    id="visualizer-select",
                    allow_blank=False,
                )
                yield Rule()
                with Vertical(id="paths"):
                    yield Label("Audio file")
                    yield Input(
                        value="",
                        placeholder="/path/to/audio.wav",
                        id="audio-path",
                    )
                    yield Label("Output video")
                    yield Input(
                        value="output_waveform.mov",
                        placeholder="/path/to/output.mov",
                        id="output-path",
                    )
                with Horizontal(id="actions"):
                    yield Button("Generate", variant="primary", id="generate")
                    yield Button("Reset config", id="reset-config")
                with Vertical(id="status-row"):
                    yield Label("Idle", id="status-label")
                    yield ProgressBar(total=100, show_eta=False, id="progress")
                yield RichLog(id="log", highlight=True, markup=True)

            viz = get_visualizer(self._current_id)
            with Vertical(id="config-pane"):
                yield Static("Config", classes="section-title")
                yield ConfigForm(
                    viz.config_schema,
                    viz.get_default_config(),
                    id="config-form",
                )
        yield Footer()

    def _log(self, message: str) -> None:
        self.query_one("#log", RichLog).write(message)

    def _set_status(self, message: str) -> None:
        self.query_one("#status-label", Label).update(message)

    def _set_progress(self, fraction: float) -> None:
        bar = self.query_one("#progress", ProgressBar)
        if fraction < 0:
            return
        bar.update(progress=max(0.0, min(100.0, fraction * 100.0)))

    def watch_busy(self, busy: bool) -> None:
        self.query_one("#generate", Button).disabled = busy
        self.query_one("#visualizer-select", Select).disabled = busy

    def _remount_config_form(self) -> None:
        viz = get_visualizer(self._current_id)
        form = self.query_one("#config-form", ConfigForm)
        form.schema = viz.config_schema
        form.defaults = viz.get_default_config()
        form.apply_defaults()

    @on(Select.Changed, "#visualizer-select")
    def on_visualizer_changed(self, event: Select.Changed) -> None:
        if event.value is Select.BLANK:
            return
        self._current_id = str(event.value)
        # Same schema shape for now; reset values when switching visualizers.
        self._remount_config_form()
        viz = get_visualizer(self._current_id)
        self._log(f"Selected visualizer: [bold]{viz.name}[/bold]")

    @on(Button.Pressed, "#reset-config")
    def on_reset_config(self) -> None:
        viz = get_visualizer(self._current_id)
        form = self.query_one("#config-form", ConfigForm)
        form.defaults = viz.get_default_config()
        form.apply_defaults()
        self._log("Config reset to defaults.")

    @on(Button.Pressed, "#generate")
    def on_generate_pressed(self) -> None:
        self.action_generate()

    def action_generate(self) -> None:
        if self.busy:
            return

        audio = self.query_one("#audio-path", Input).value.strip()
        output = self.query_one("#output-path", Input).value.strip()
        if not audio:
            self._set_status("Choose an audio file path.")
            self._log("[red]Audio path is required.[/red]")
            return
        if not output:
            self._set_status("Choose an output path.")
            self._log("[red]Output path is required.[/red]")
            return
        if not Path(audio).expanduser().exists():
            self._set_status("Audio file not found.")
            self._log(f"[red]File not found:[/red] {audio}")
            return

        try:
            config = self.query_one("#config-form", ConfigForm).read_config()
        except ValueError as exc:
            self._set_status("Invalid config.")
            self._log(f"[red]Config error:[/red] {exc}")
            return

        self.busy = True
        self._set_progress(0.0)
        self._set_status("Starting…")
        self._log(f"Generating [bold]{get_visualizer(self._current_id).name}[/bold] → {output}")
        self.run_generate(audio, output, config)

    @work(thread=True)
    def run_generate(self, audio: str, output: str, config: dict[str, Any]) -> None:
        viz = get_visualizer(self._current_id)

        def on_progress(fraction: float, message: str) -> None:
            self.call_from_thread(self._set_status, message)
            if fraction >= 0:
                self.call_from_thread(self._set_progress, fraction)
                # Avoid flooding the log; status + bar already track fine-grained %.
                if fraction in (0.0, 1.0) or abs(fraction - getattr(self, "_last_logged_frac", -1)) >= 0.1:
                    self._last_logged_frac = fraction
                    self.call_from_thread(self._log, f"{message} ({fraction:.0%})")
            else:
                self.call_from_thread(self._log, message)

        try:
            result = viz.run(
                audio_path=str(Path(audio).expanduser()),
                output_path=str(Path(output).expanduser()),
                config=config,
                progress_callback=on_progress,
            )
        except Exception as exc:  # noqa: BLE001 - surface any render failure in the TUI
            tb = traceback.format_exc()
            self.call_from_thread(self._log, f"[red]Error:[/red] {exc}\n{tb}")
            self.call_from_thread(self._set_status, f"Failed: {exc}")
            self.call_from_thread(self._set_busy, False)
            return

        self.call_from_thread(self._set_progress, 1.0)
        self.call_from_thread(self._set_status, f"Done → {result}")
        self.call_from_thread(self._log, f"[green]Wrote[/green] {result}")
        self.call_from_thread(self._set_busy, False)

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy


def main() -> None:
    VisualizerApp().run()


if __name__ == "__main__":
    main()
