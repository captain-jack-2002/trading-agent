from abc import ABC, abstractmethod

from trading_agent.models.domain import OrderRequest, OrderResult


class BrokerExecutionAdapter(ABC):
    @abstractmethod
    def submit(self, order: OrderRequest) -> OrderResult:
        """Must evaluate authoritative risk state before any execution."""
