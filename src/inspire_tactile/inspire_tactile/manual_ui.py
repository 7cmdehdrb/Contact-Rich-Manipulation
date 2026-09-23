"""Isaac Sim native panel. Import/construct only after AppLauncher starts."""

import math


class JointPanel:
    def __init__(self, controls):
        import omni.ui as ui

        self.controls = controls
        self.actual = dict(controls.commanded)
        self.updating = False
        self.models, self.rows, self.readouts = {}, {}, {}
        self.window = ui.Window(
            "Inspire Hand - Manual Joints", width=620,
            height=min(720, max(400, ui.Workspace.get_main_window_height() - 140)),
            position_x=max(20, ui.Workspace.get_main_window_width() - 650), position_y=80,
        )
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("12 individual joints | angles in DEGREES", height=24)
                ui.Label("Arm stays fixed. Contact can prevent reaching the requested angle.", height=22)
                with ui.HStack(height=26):
                    self.link_model = ui.SimpleBoolModel(controls.linked)
                    ui.CheckBox(model=self.link_model, width=25)
                    ui.Label("Link original mimic joints (6 motor targets)")
                self.link_model.add_value_changed_fn(self._linked_changed)
                with ui.HStack(height=30, spacing=5):
                    ui.Button("Open", clicked_fn=lambda: self._preset("open"))
                    ui.Button("Close", clicked_fn=lambda: self._preset("close"))
                    ui.Button("Hold actual", clicked_fn=self._hold)
                    ui.Button("Reset scene", clicked_fn=self._reset)
                ui.Label("Independent mode is a simulation-only 12-DOF override.", height=22)
                with ui.ScrollingFrame():
                    with ui.VStack(spacing=4):
                        for name in controls.names:
                            spec = controls.specs[name]
                            with ui.VStack(height=42):
                                with ui.HStack(height=18):
                                    short = name.removeprefix("inspire_left_").removesuffix("_joint")
                                    suffix = " [mimic]" if spec["mimic"] else ""
                                    ui.Label(short + suffix, width=185)
                                    self.readouts[name] = ui.Label("", width=320)
                                with ui.HStack(height=24) as row:
                                    self.rows[name] = row
                                    low, high = math.degrees(spec["lower"]), math.degrees(spec["upper"])
                                    model = ui.SimpleFloatModel(math.degrees(controls.desired[name]), min=low, max=high)
                                    self.models[name] = model
                                    ui.FloatSlider(model=model, min=low, max=high, step=0.1)
                                    ui.FloatField(model=model, width=80)
                                    model.add_value_changed_fn(lambda m, n=name: self._changed(n, m))
                self.status = ui.Label("Waiting for sensor data", height=28, word_wrap=True)
        self.sync()

    def _changed(self, name, model):
        if not self.updating:
            try:
                self.controls.set_target(name, math.radians(model.as_float))
            except ValueError as exc:
                self.status.text = str(exc)
            finally:
                self.sync()

    def _linked_changed(self, model):
        self.controls.set_linked(model.as_bool)
        self.sync()

    def _preset(self, name):
        self.controls.preset(name)
        self.sync()

    def _hold(self):
        self.controls.hold(self.actual)
        self.sync()

    def _reset(self):
        self.controls.reset_requested = True

    def sync(self):
        self.updating = True
        try:
            for name, model in self.models.items():
                model.set_value(math.degrees(self.controls.desired[name]))
                self.rows[name].enabled = not (self.controls.linked and self.controls.specs[name]["mimic"])
        finally:
            self.updating = False

    def refresh(self, actual, bits, time_s):
        self.actual = actual
        for name, label in self.readouts.items():
            label.text = (f"command {math.degrees(self.controls.commanded[name]):6.1f} deg | "
                          f"actual {math.degrees(actual[name]):6.1f} deg")
        active = [str(i) for i, bit in enumerate(bits) if bit]
        self.status.text = f"Sim {time_s:.2f} s | Tactile ON IDs: {', '.join(active) or 'none'}"

    def destroy(self):
        self.window.destroy()
