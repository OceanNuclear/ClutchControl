from typing import Callable, Iterable

import tkinter as tk
from tkinter import ttk
import numpy as np

FRAME_RATE = 60
FordFiesta_ratio = {}  # wheel diameter * gearing ratio
motorcycle_ratio = {
    1:8/8000,
    2:20/8000,
    3:38/8000,
    4:55/8000,
    5:61/8000,
}
motorcycle_friction_coefs = {
    "wheel_speed": [0.0005, 0, 5],
    "crank_rpm": [0.5/10000, 0.5, 300],  # https://www.desmos.com/calculator/mwtoacy0bl
    "clutch": [25000, 15000]
}


class DriveTrainSimulator:

    def __init__(self, root,
            drive_ratio: dict[int, float],
            friction_coef_per_second: dict[str, tuple[float, float]],
            throttle_curve: Callable[[float], float],
            frame_rate=FRAME_RATE
        ):
        # init root
        self.root = root
        self.root.title("Linear driving simulator")
        self.root.geometry("550x450")
        self.root.configure(bg="white")
        self.frame_dt = round(1000/frame_rate)*0.001

        # init values
        self.drive_ratio = drive_ratio
        self.friction_coefs = {friction_name: [c*self.frame_dt for c in coef_tuple] for friction_name, coef_tuple in friction_coef_per_second.items()}
        self.throttle_curve = throttle_curve
        self.wheel_speed = 30  # mph
        self.crank_rpm = 9000  # hundred-RPMs
        self.throttle_openness = 0.11  # 11 = default idle. 0 = ignition off.
        self.clutch_lift = 1
        self.gear = 3
        self.rev_limiter_on = False

        # Canvas to hold everything
        self.canvas = tk.Canvas(root, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.sliders = {
            "speedometer": ttk.Scale(root, from_=180, to=0, orient="vertical", length=250),
            "tachometer": ttk.Scale(root, from_=12000, to=0, orient="vertical", length=250),
            "throttle": ttk.Scale(root, from_=1, to=0, orient="vertical", length=250),
            "clutch": ttk.Scale(root, from_=0, to=1, orient="horizontal", length=400,
                command=self.control_clutch),
        }
        self.sliders["speedometer"].set(self.wheel_speed)
        self.sliders["tachometer"].set(self.crank_rpm)
        self.sliders["throttle"].set(self.throttle_openness)
        self.sliders["clutch"].set(self.clutch_lift)

        # Place vertical sliders on canvas
        self.win_idx = {
            "speedometer": self.canvas.create_window(120, 150, window=self.sliders["speedometer"]),
            "tachometer": self.canvas.create_window(280, 150, window=self.sliders["tachometer"]),
            "throttle": self.canvas.create_window(440, 150, window=self.sliders["throttle"]),
            "clutch": self.canvas.create_window(250, 350, window=self.sliders["clutch"]),
        }

        # Labels for vertical sliders
        self.labels = {
            "speedometer": tk.Label(root, text="Wheel speed\n(mph)", bg="white", font=("Arial", 10, "bold")),
            "tachometer": tk.Label(root, text="Tachometer\n(RPM)", bg="white", font=("Arial", 10, "bold")),
            "throttle": tk.Label(root, text="Throttle\nposition", bg="white", font=("Arial", 10, "bold")),
            "clutch": tk.Label(root, text="Clutch", bg="white", font=("Arial", 10, "bold"))
        }
        self.win_idx.update({
            "speedometer_label": self.canvas.create_window(120, 20, window=self.labels["speedometer"]),
            "tachometer_label": self.canvas.create_window(280, 20, window=self.labels["tachometer"]),
            "throttle_label": self.canvas.create_window(440, 20, window=self.labels["throttle"]),
            "clutch_label": self.canvas.create_window(200, 320, window=self.labels["clutch"])
        })

        # create key bindings
        for i in range(10):
            self.root.bind(str(i), self.open_throttle)
        self.root.bind('<Tab>', self.shift_down)
        self.root.bind('<ISO_Left_Tab>', self.shift_up)

        # Line connecting vertical sliders
        self.lines = {
            "speedometer": self.canvas.create_line(0, 0, 0, 0, width=2, fill="blue"),
            "tachometer": self.canvas.create_line(0, 0, 0, 0, width=2, fill="blue"),
            "throttle": self.canvas.create_line(0, 0, 0, 0, width=2, fill="blue"),
        }
        print(f"{self.crank_rpm=}, {self.wheel_speed=}")
        self.take_time_step()

    def take_time_step(self):
        """Take one step forward in time."""
        ## physics calculations
        current_conversion_ratio = self.drive_ratio[self.gear]
        # rev limiter
        if self.crank_rpm > 10500:
            self.rev_limiter_on = True
        elif self.crank_rpm < 9500:  # hysteresis needed
            self.rev_limiter_on = False

        self.ideal_wheel_speed = current_conversion_ratio * self.crank_rpm
        self.ideal_crank_rpm = self.wheel_speed / current_conversion_ratio
        self.ideal_crank_rpm_throttle = self.find_equilibrium_rpm(self.throttle_openness)
        # Calculate the accumulated acceleration (shortened to acc.) on individual
        # components, before accounting for the clutch.
        wheel_external_acc = - self.rolling_resistance(
            self.wheel_speed, self.friction_coefs["wheel_speed"]
        )  # ignore braking for now.
        crank_external_acc = (
            - self.rolling_resistance(self.crank_rpm, self.friction_coefs["crank_rpm"])
            + self.throttle_curve(self.crank_rpm) * self.throttle_openness * (1-int(self.rev_limiter_on))
        )
        # print(f"{self.wheel_speed=}, {wheel_external_acc=}, {self.crank_rpm=}, {crank_external_acc=}")
        # Accounting for clutch now.
        is_slipping = ~np.isclose(self.ideal_crank_rpm, self.crank_rpm)
        if is_slipping:
            sign = np.sign(self.ideal_crank_rpm - self.crank_rpm)
            clutch_acc_on_crank = self.clutch_force(is_slipping)
            crank_rpm_update = crank_external_acc + sign * clutch_acc_on_crank
            wheel_speed_update = wheel_external_acc - sign * clutch_acc_on_crank * current_conversion_ratio
        else:
            acc_difference_on_crank = wheel_external_acc/current_conversion_ratio - crank_external_acc
            sign = np.sign(acc_difference_on_crank) # crank RPM increase if positive.
            max_clutch_acc_on_crank = self.clutch_force(is_slipping)  # not slipping yet.
            if abs(acc_difference_on_crank)> max_clutch_acc_on_crank:  # begin slipping.
                max_clutch_acc_on_wheel = max_clutch_acc_on_crank * current_conversion_ratio

                crank_rpm_update = crank_external_acc - sign * max_clutch_acc_on_crank
                wheel_speed_update = wheel_external_acc - sign * max_clutch_acc_on_wheel
            else:  # stays attached
                crank_rpm_update = crank_external_acc + wheel_external_acc / current_conversion_ratio
                wheel_speed_update = wheel_external_acc + crank_external_acc * current_conversion_ratio

        # Pause briefly when the ideal speed v.s. actual speeds pass each other.
        step_size = 1.0
        new_crank_rpm = self.crank_rpm + crank_rpm_update
        new_ideal_crank_rpm = (self.wheel_speed + wheel_speed_update) / current_conversion_ratio
        ideal_crank_rpm_change = new_ideal_crank_rpm - self.ideal_crank_rpm
        if abs(np.sign(new_ideal_crank_rpm - new_crank_rpm) - np.sign(self.ideal_crank_rpm - self.crank_rpm)) == 2:
            # crossover has occurred
            step_size = (self.ideal_crank_rpm - self.crank_rpm)/(crank_rpm_update - ideal_crank_rpm_change) # should be less than 1.
            if step_size>1 or step_size<0:
                raise ValueError("Programmer error")

        self.crank_rpm = np.clip(self.crank_rpm + step_size * crank_rpm_update, 0, np.inf)
        self.wheel_speed = np.clip(self.wheel_speed + step_size * wheel_speed_update, 0, np.inf)
        self.sliders["tachometer"].set(self.crank_rpm)
        self.sliders["speedometer"].set(self.wheel_speed)
        slip_status = "slipping" if is_slipping else "no slip-"
        print(f"{slip_status}, step_size_taken={step_size}, wheel_speed={self.wheel_speed}, crank_rpm={self.crank_rpm}")

        # Changes to the clutch engagement causes the engine RPM to get pulled down more/less.
        speedo_x = (
            self.canvas.coords(self.win_idx["speedometer"])[0]
            + self.sliders["speedometer"].winfo_width()/2
        )
        ideal_speed_y = self.get_y_corresponding_to_value(
            self.sliders["speedometer"], self.win_idx["speedometer"], self.ideal_wheel_speed
        )
        speed_y = self.get_y_corresponding_to_value(
            self.sliders["speedometer"], self.win_idx["speedometer"], self.wheel_speed
        )

        tacho_lx = self.canvas.coords(self.win_idx["tachometer"])[0] - self.sliders["tachometer"].winfo_width()/2
        ideal_crank_y = self.get_y_corresponding_to_value(
            self.sliders["tachometer"], self.win_idx["tachometer"], self.ideal_crank_rpm
        )
        crank_y = self.get_y_corresponding_to_value(
            self.sliders["tachometer"], self.win_idx["tachometer"], self.crank_rpm
        )

        tacho_rx = tacho_lx + self.sliders["tachometer"].winfo_width()/2
        ideal_crank_y_throttle = self.get_y_corresponding_to_value(
            self.sliders["tachometer"], self.win_idx["tachometer"], self.ideal_crank_rpm_throttle
        )
        throttle_x = self.canvas.coords(self.win_idx["throttle"])[0] - self.sliders["throttle"].winfo_width()/2
        throttle_y = self.get_y_corresponding_to_value(
            self.sliders["throttle"], self.win_idx["throttle"], self.throttle_openness
        )

        self.canvas.coords(self.lines["speedometer"], tacho_lx, crank_y, speedo_x, ideal_speed_y)
        self.canvas.coords(self.lines["tachometer"], speedo_x, speed_y, tacho_lx, ideal_crank_y)
        self.canvas.coords(self.lines["throttle"], throttle_x, throttle_y, tacho_rx, ideal_crank_y_throttle)

        # Turn red for lugging/stalling engine
        if round(self.crank_rpm) < 1200:
            self.set_background("#FF3133")
        else:
            self.set_background("white")

        self.root.after(int(self.frame_dt*1000), self.take_time_step)

    def open_throttle(self, event):
        """Change the throttle position"""
        self.throttle_openness = int(event.char) * 0.11  # Map 0-9 to 0-99
        self.sliders["throttle"].set(self.throttle_openness)

    def control_clutch(self, clutch_lift):
        """Change how disengaged the clutch is."""
        self.clutch_lift = float(clutch_lift)
        # self.sliders["clutch"].set(self.clutch_lift)

    def clutch_force(self, is_slipping: bool):
        """Return the absolute value of the clutch's restoring force on the engine."""
        return self.friction_coefs["clutch"][int(is_slipping)] * (1 - self.clutch_lift)

    def shift_up(self, event):
        """Shift up a gear."""
        self.gear = min([self.gear+1, max(self.drive_ratio)])
        print(f"Shifted up to {self.gear}.")

    def shift_down(self, event):
        """Shift up a gear."""
        self.gear = max([self.gear-1, min(self.drive_ratio)])
        print(f"Shifted down to {self.gear}.")

    def find_equilibrium_rpm(self, throttle_openness):
        return 0.0

    def get_y_corresponding_to_value(
            self, slider: ttk.Scale, window_id: int, test_value: float
        ):
        """Approximate vertical center of slider handle in canvas coordinates."""
        slider.update_idletasks()
        x, y = self.canvas.coords(window_id)
        length = int(slider['length'])
        range_ = abs(slider["to"] - slider["from"])
        handle_y = y + (length / 2) - (test_value / range_) * length
        return handle_y

    @staticmethod
    def rolling_resistance(speed: float, poly_coefs: Iterable[float]) -> float:
        """
        Calculate a rolling friction coefficient, which follows a
        simple upside-down-parabolic relationship with speed.
        If an object is allowed to coast to zero velocity under this rolling coefficient
        profile, its speed will resemble a negative exponential curve.
        """
        return np.poly1d(poly_coefs)(speed)

    def set_background(self, color):
        """set the background colour of the window to a string."""
        self.root.configure(bg=color)
        self.canvas.configure(bg=color)
        for label_type in self.labels.values():
            label_type.configure(bg=color)


if __name__ == "__main__":
    motorcycle_throttle_curve = lambda x: -0.000000088*(x**2) * (x-13000) + 220 * abs(np.sqrt(x))
    app = DriveTrainSimulator(tk.Tk(), motorcycle_ratio, motorcycle_friction_coefs, motorcycle_throttle_curve)
    app.root.mainloop()
    # app.take_time_step()