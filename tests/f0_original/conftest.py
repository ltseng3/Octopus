import pytest
from octopus_fusion import AgentEvidence, FusionObservation, ReliabilityEstimate


@pytest.fixture
def observation():
    return FusionObservation('f0:r0','f0',50.0,(
        AgentEvidence('A',1,'f0:rA','A',45.0,0.95),
        AgentEvidence('B',0,'f0:rB','B',45.0,0.15),
        AgentEvidence('C',1,'f0:rC','C',45.0,0.90),
    ))


@pytest.fixture
def reliability():
    return ReliabilityEstimate((('A',.05),('B',.35),('C',.10)),'hypothetical_test')
