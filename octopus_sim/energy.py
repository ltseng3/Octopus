"""Optional, deterministic, device-attributed energy accounting.

Energy is calculated from *actual service intervals*, never request submission
or message arrival timestamps. Modelled baseline power is charged once per
physical device over a common observation horizon. Incremental work performed
after that horizon is retained as explicitly reported post-window energy.

Two alternative, mutually exclusive radio modes are provided per device:
``per_bit`` and ``state`` (incremental TX/RX watts during link serialization).
All values in this module are joules, watts, bits or integer microseconds.
This is accounting, not a physical battery or thermal simulator.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class _ComputeStart:
    device: str
    model_id: str
    start_us: int
    incremental_w: float
    incremental_j_per_execution: float | None


@dataclass(frozen=True)
class _TransferStart:
    sender: str
    receiver: str
    bytes_sent: int
    role: str
    start_us: int


class EnergyLedger:
    """Post-processes actual service intervals with strictly defined accounting.

    `window_us` is a common, policy-independent decision horizon, computed
    from scheduled event releases and the decision deadline or given explicitly.
    Queue wait and propagation do not contribute *active* power. Baseline power
    includes radio-idle power and is charged exactly once per physical device.
    """

    def __init__(self, config: Mapping, agents: tuple[str, ...], window_us: int,
                 network_channels: int):
        if window_us <= 0:
            raise ValueError('Energy measurement window must be positive')
        self.config = config
        self.window_us = window_us
        self.devices: dict[str, dict] = {'edge': config['device_profiles']['edge']}
        for a in agents:
            self.devices[a] = config['agent_overrides'].get(a, config['device_profiles']['agent'])
        if any(v['radio']['mode'] == 'state' for v in self.devices.values()) and network_channels != 1:
            raise ValueError('state-based radio model requires network_channels=1 (half-duplex proxy)')
        self._computing: dict[str, _ComputeStart] = {}
        self._transferring: dict[str, _TransferStart] = {}
        self.by_device: dict[str, dict[str, float]] = {
            device: dict(baseline_j=0.0, compute_j=0.0, radio_tx_j=0.0,
                         radio_rx_j=0.0, window_incremental_j=0.0,
                         post_window_incremental_j=0.0)
            for device in self.devices
        }
        self.by_model: dict[str, float] = defaultdict(float)
        self.by_transfer_role: dict[str, float] = defaultdict(float)
        self._finalized = False

    @staticmethod
    def _split(start_us: int, end_us: int, joules: float, window_us: int) -> tuple[float, float]:
        if end_us < start_us or start_us < 0 or joules < 0:
            raise ValueError('Invalid energy interval')
        if end_us == start_us:
            return (joules, 0.0) if start_us <= window_us else (0.0, joules)
        window_time = max(0, min(end_us, window_us) - min(start_us, window_us))
        first = min(joules, max(0.0, joules * window_time / (end_us - start_us)))
        return first, max(0.0, joules - first)

    def _book(self, device: str, component: str, joules: float,
              start_us: int, end_us: int) -> None:
        if self._finalized:
            raise RuntimeError('Energy ledger already finalized')
        window_j, post_j = self._split(start_us, end_us, joules, self.window_us)
        target = self.by_device[device]
        target[component] += joules
        target['window_incremental_j'] += window_j
        target['post_window_incremental_j'] += post_j

    def begin_compute(self, token: str, device: str, model_id: str, start_us: int) -> None:
        if token in self._computing:
            raise RuntimeError(f'Duplicate compute token {token}')
        power = self.config['model_incremental_w'].get(
            model_id, self.devices[device]['compute_incremental_w'])
        self._computing[token] = _ComputeStart(
            device, model_id, start_us, power, self.config.get('model_energy_j', {}).get(model_id))

    def end_compute(self, token: str, end_us: int) -> None:
        try:
            rec = self._computing.pop(token)
        except KeyError as exc:
            raise RuntimeError(f'Missing compute start: {token}') from exc
        if end_us < rec.start_us:
            raise RuntimeError('Compute finished before starting')
        joules = (rec.incremental_j_per_execution if rec.incremental_j_per_execution is not None
                  else (end_us - rec.start_us) / 1_000_000 * rec.incremental_w)
        self._book(rec.device, 'compute_j', joules, rec.start_us, end_us)
        self.by_model[rec.model_id] += joules

    def begin_transfer(self, token: str, sender: str, receiver: str,
                       bytes_sent: int, role: str, start_us: int) -> None:
        if token in self._transferring:
            raise RuntimeError(f'Duplicate transfer token {token}')
        self._transferring[token] = _TransferStart(sender, receiver, bytes_sent, role, start_us)

    def end_transfer(self, token: str, end_us: int) -> None:
        try:
            rec = self._transferring.pop(token)
        except KeyError as exc:
            raise RuntimeError(f'Missing transfer start: {token}') from exc
        if end_us < rec.start_us:
            raise RuntimeError('Transfer finished before starting')
        total = 0.0
        for device, action, component in ((rec.sender, 'tx', 'radio_tx_j'),
                                          (rec.receiver, 'rx', 'radio_rx_j')):
            radio = self.devices[device]['radio']
            if radio['mode'] == 'per_bit':
                joules = 8 * rec.bytes_sent * radio[f'{action}_j_per_bit']
            else:
                joules = (end_us - rec.start_us) / 1_000_000 * radio[f'{action}_incremental_w']
            self._book(device, component, joules, rec.start_us, end_us)
            total += joules
        self.by_transfer_role[rec.role] += total

    def finalize(self) -> dict:
        if self._finalized or self._computing or self._transferring:
            raise RuntimeError('Energy ledger cannot finalize with open jobs or twice')
        self._finalized = True
        for device, profile in self.devices.items():
            self.by_device[device]['baseline_j'] = profile['baseline_w'] * self.window_us / 1_000_000
        per_device = {}
        totals = defaultdict(float)
        for device, entry in self.by_device.items():
            row = dict(entry)
            row['radio_j'] = row['radio_tx_j'] + row['radio_rx_j']
            row['incremental_j'] = row['compute_j'] + row['radio_j']
            row['window_total_j'] = row['baseline_j'] + row['window_incremental_j']
            row['total_j'] = row['baseline_j'] + row['incremental_j']
            per_device[device] = row
            for k, value in row.items():
                totals[k] += value
        results = {
            'energy_enabled': True,
            'energy_calibration': self.config['calibration'],
            'energy_calibration_source': self.config.get('calibration_source'),
            'energy_measurement_window_ms': self.window_us / 1000,
            'energy_baseline_j': totals['baseline_j'],
            'energy_compute_incremental_j': totals['compute_j'],
            'energy_radio_tx_incremental_j': totals['radio_tx_j'],
            'energy_radio_rx_incremental_j': totals['radio_rx_j'],
            'energy_radio_incremental_j': totals['radio_j'],
            'energy_workload_incremental_j': totals['incremental_j'],
            'energy_window_total_j': totals['window_total_j'],
            'energy_post_window_incremental_j': totals['post_window_incremental_j'],
            'energy_total_j': totals['total_j'],
            'energy_vehicle_total_j': sum(v['total_j'] for k,v in per_device.items() if k != 'edge'),
            'energy_edge_total_j': per_device['edge']['total_j'],
            'energy_by_device_j': per_device,
            'energy_compute_by_model_j': dict(sorted(self.by_model.items())),
            'energy_radio_by_transfer_role_j': dict(sorted(self.by_transfer_role.items())),
            'energy_accounting_boundary': 'device baseline over fixed horizon; all executed incremental work including drain',
        }
        return results
