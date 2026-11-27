#!/usr/bin/env bash
# Pi 5 active cooler diagnostics. The fan is PWM-controlled; if device tree shows
# cooling_fan status=disabled, firmware did not enable the driver at boot.
set -euo pipefail

status_path="/proc/device-tree/cooling_fan/status"
temp="$(vcgencmd measure_temp 2>/dev/null || true)"
throttle="$(vcgencmd get_throttled 2>/dev/null || true)"

echo "Temperature: ${temp}"
echo "Throttled:   ${throttle}"

if [[ -r "${status_path}" ]]; then
  fan_status="$(tr -d '\000' < "${status_path}")"
  echo "cooling_fan device-tree status: ${fan_status}"
else
  echo "cooling_fan device-tree node not found (not a Pi 5 fan setup?)."
  fan_status="missing"
fi

echo
if [[ "${fan_status}" == "disabled" ]]; then
  echo "The Pi 5 fan driver is DISABLED. Symptom: fan may twitch at boot, then stay off"
  echo "even when the SoC is hot (~80°C+), which causes throttling during baseline runs."
  echo
  echo "Fix (official or JST PWM fan on the 4-pin header next to GPIO):"
  echo "  1. Ensure the fan is on the board fan connector (not GPIO alone unless wired for PWM)."
  echo "  2. Add to /boot/firmware/config.txt under [pi5] (see docs/pi5-active-cooler.snippet), then reboot"
  echo "  4. After reboot, re-run: ./scripts/pi_cooling_check.sh"
  echo
  echo "One-shot enable without editing config (until reboot):"
  echo "  sudo dtparam cooling_fan=on"
elif [[ "${fan_status}" == "okay" ]]; then
  echo "Fan driver is enabled. Below 50°C the fan is normally off; above 50°C it ramps up."
else
  echo "See Raspberry Pi 5 active cooler docs if the fan still does not spin when hot."
fi