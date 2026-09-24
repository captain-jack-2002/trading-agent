from trading_agent.execution.base import BrokerExecutionAdapter
from trading_agent.models.domain import OrderRequest, OrderResult


class HDFCSkyAdapter(BrokerExecutionAdapter):
    """Deliberately unavailable until official specifications and live safety review."""

    def submit(self, order: OrderRequest) -> OrderResult:
        raise NotImplementedError("Live HDFC SKY execution is disabled in Phase 1")
