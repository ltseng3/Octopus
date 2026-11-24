from pathlib import Path
import pytest
from octopus_sim.config import ExperimentConfig


@pytest.fixture
def root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def build():
    def create(policy='always_share', agents=('a', 'b'), episodes=4, **kwargs):
        info = {
            'scenario': {
                'agents': list(agents), 'episodes': episodes,
                'arrival_interval_ms': 200, 'deadline_ms': 125,
                'seed': kwargs.pop('seed', 61),
            },
            'world': {
                'local_error': kwargs.pop('local_error', .10),
                'edge_error': kwargs.pop('edge_error', .06),
                'edge_correlation': kwargs.pop('edge_correlation', 0.0),
                'local_correlation': kwargs.pop('local_correlation', 0.0),
            },
            'resources': {
                'edge_workers': kwargs.pop('edge_workers', 2),
                'edge_inference_ms': kwargs.pop('edge_ms', 30),
                'network_latency_ms': kwargs.pop('network_ms', 3),
                'observation_bytes': kwargs.pop('observation_bytes', 1024),
                'result_bytes': kwargs.pop('result_bytes', 128),
                'packet_loss': kwargs.pop('packet_loss', 0.0),
            },
            'policy': {'name': policy, **kwargs.pop('policy_args', {})},
        }
        if kwargs:
            raise AssertionError(f'Unused fixture keys {kwargs}')
        return ExperimentConfig.from_dict(info)
    return create
