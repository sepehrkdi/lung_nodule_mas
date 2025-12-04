"""
Communication Module
====================

Inter-agent communication infrastructure.

EDUCATIONAL PURPOSE - AGENT COMMUNICATION:
- Speech Acts: Performatives (inform, achieve, query)
- FIPA-ACL: Foundation for Intelligent Physical Agents standards
- Message Passing: Asynchronous agent coordination

Key Concepts Demonstrated:
- Communication as Action: Messages change agent mental states
- Illocutionary Force: The intent behind the message
- Source Tracking: Belief provenance via [source(agent)]
"""

from .message_queue import Message, MessageBroker, Performative

__all__ = ['Message', 'MessageBroker', 'Performative']
