"""
Analysis State Manager
======================

Manages the state of ongoing and completed analyses for polling-based updates.
Stores agent results as they complete, allowing the UI to poll for progress.
"""

import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from a single agent."""
    agent_name: str
    agent_type: str
    approach: str
    weight: float
    probability: float
    predicted_class: int
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AnalysisState:
    """State of an ongoing or completed analysis."""
    session_id: str
    nodule_id: str
    status: str  # "pending", "running", "completed", "error"
    total_agents: int = 6  # 3 radiologists + 3 pathologists
    completed_agents: List[AgentResult] = field(default_factory=list)
    consensus: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def completed_count(self) -> int:
        return len(self.completed_agents)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "nodule_id": self.nodule_id,
            "status": self.status,
            "total_agents": self.total_agents,
            "completed_count": self.completed_count,
            "completed_agents": [
                {
                    "agent_name": a.agent_name,
                    "agent_type": a.agent_type,
                    "approach": a.approach,
                    "weight": a.weight,
                    "probability": a.probability,
                    "predicted_class": a.predicted_class,
                    "details": a.details,
                    "timestamp": a.timestamp,
                }
                for a in self.completed_agents
            ],
            "consensus": self.consensus,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


class AnalysisStateManager:
    """
    Thread-safe manager for analysis states.
    
    Stores ongoing and completed analysis sessions, allowing:
    - Starting new analyses
    - Adding agent results as they complete
    - Setting final consensus
    - Polling for current state
    """
    
    def __init__(self, max_sessions: int = 100):
        """
        Initialize the state manager.
        
        Args:
            max_sessions: Maximum number of sessions to keep in memory
        """
        self._sessions: Dict[str, AnalysisState] = {}
        self._lock = Lock()
        self._max_sessions = max_sessions
    
    def start_analysis(self, nodule_id: str) -> str:
        """
        Start a new analysis session.
        
        Args:
            nodule_id: The nodule ID being analyzed
            
        Returns:
            session_id: Unique identifier for this analysis session
        """
        session_id = str(uuid.uuid4())
        
        with self._lock:
            # Cleanup old sessions if at capacity
            if len(self._sessions) >= self._max_sessions:
                self._cleanup_old_sessions()
            
            self._sessions[session_id] = AnalysisState(
                session_id=session_id,
                nodule_id=nodule_id,
                status="running"
            )
        
        logger.info(f"Started analysis session {session_id} for nodule {nodule_id}")
        return session_id
    
    def add_agent_result(
        self,
        session_id: str,
        agent_name: str,
        result: Dict[str, Any]
    ) -> bool:
        """
        Add a completed agent's result to the session.
        
        Args:
            session_id: The analysis session ID
            agent_name: Name of the agent that completed
            result: The agent's result dictionary
            
        Returns:
            True if successful, False if session not found
        """
        with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            state = self._sessions[session_id]
            
            # Extract agent info from result
            findings = result.get("findings", {})
            # Use None check instead of 'or' to handle 0.0 correctly
            prob = findings.get("malignancy_probability")
            if prob is None:
                prob = findings.get("text_malignancy_probability")
            if prob is None:
                prob = 0.5
            
            agent_result = AgentResult(
                agent_name=agent_name,
                agent_type=result.get("_agent_type", "unknown"),
                approach=result.get("approach", "unknown"),
                weight=result.get("weight", 1.0),
                probability=prob,
                predicted_class=findings.get("predicted_class", 3),
                details=findings,
            )
            
            state.completed_agents.append(agent_result)
            state.updated_at = datetime.now().isoformat()
            
            logger.debug(
                f"Session {session_id}: Agent {agent_name} completed "
                f"({state.completed_count}/{state.total_agents})"
            )
            
            return True
    
    def set_consensus(
        self,
        session_id: str,
        consensus_result: Dict[str, Any]
    ) -> bool:
        """
        Set the final consensus result for a session.
        
        Args:
            session_id: The analysis session ID
            consensus_result: The consensus result dictionary
            
        Returns:
            True if successful, False if session not found
        """
        with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            state = self._sessions[session_id]
            state.consensus = consensus_result
            state.status = "completed"
            state.updated_at = datetime.now().isoformat()
            
            logger.info(f"Session {session_id} completed with consensus")
            
            return True
    
    def set_error(self, session_id: str, error_message: str) -> bool:
        """
        Mark a session as failed with an error.
        
        Args:
            session_id: The analysis session ID
            error_message: The error message
            
        Returns:
            True if successful, False if session not found
        """
        with self._lock:
            if session_id not in self._sessions:
                return False
            
            state = self._sessions[session_id]
            state.status = "error"
            state.error_message = error_message
            state.updated_at = datetime.now().isoformat()
            
            logger.error(f"Session {session_id} failed: {error_message}")
            
            return True
    
    def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of an analysis session.
        
        Args:
            session_id: The analysis session ID
            
        Returns:
            State dictionary or None if not found
        """
        with self._lock:
            if session_id not in self._sessions:
                return None
            
            return self._sessions[session_id].to_dict()
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a completed session.
        
        Args:
            session_id: The analysis session ID
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
    
    def _cleanup_old_sessions(self):
        """Remove oldest completed sessions to make room for new ones."""
        completed = [
            (sid, state) for sid, state in self._sessions.items()
            if state.status in ("completed", "error")
        ]
        
        # Sort by updated_at and remove oldest 20%
        completed.sort(key=lambda x: x[1].updated_at)
        to_remove = completed[:max(1, len(completed) // 5)]
        
        for sid, _ in to_remove:
            del self._sessions[sid]
        
        logger.info(f"Cleaned up {len(to_remove)} old sessions")


# Global instance
analysis_manager = AnalysisStateManager()
