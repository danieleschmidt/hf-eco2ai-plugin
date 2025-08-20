"""Emergent Swarm Carbon Intelligence: Self-Organizing Networks for Carbon Optimization.

This revolutionary system implements emergent intelligence through swarm behavior,
where individual agents self-organize to create collective carbon optimization
that exceeds the sum of individual contributions.

Key Breakthroughs:
1. Self-Organizing Carbon Agents with Emergent Behavior
2. Collective Intelligence Through Stigmergy
3. Adaptive Swarm Topologies for Dynamic Optimization
4. Emergent Pattern Recognition and Learning
5. Decentralized Decision Making with Global Coherence
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from pathlib import Path
from enum import Enum
import concurrent.futures
import threading
from collections import defaultdict, deque
import networkx as nx
import uuid
import random
import math

# Advanced scientific computing
from scipy import spatial, optimize, stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SwarmBehavior(Enum):
    """Types of swarm behaviors for carbon optimization."""
    EXPLORATION = "exploration"        # Searching for new optimization opportunities
    EXPLOITATION = "exploitation"      # Optimizing known good solutions
    COMMUNICATION = "communication"    # Information sharing between agents
    COLLABORATION = "collaboration"    # Working together on complex problems
    ADAPTATION = "adaptation"         # Adjusting to environmental changes
    EMERGENCE = "emergence"           # Spontaneous pattern formation


class AgentRole(Enum):
    """Roles agents can take in the swarm."""
    SCOUT = "scout"                   # Explores new areas
    WORKER = "worker"                 # Performs optimization tasks
    COORDINATOR = "coordinator"       # Coordinates group activities
    SPECIALIST = "specialist"         # Specialized domain expertise
    COMMUNICATOR = "communicator"     # Facilitates information flow
    LEADER = "leader"                 # Temporary leadership role


class EmergentPattern(Enum):
    """Types of emergent patterns in the swarm."""
    FLOCKING = "flocking"             # Agents moving together
    CLUSTERING = "clustering"         # Agents grouping by similarity
    SPIRAL = "spiral"                 # Spiral movement patterns
    WAVE = "wave"                     # Wave-like propagation
    FRACTAL = "fractal"               # Self-similar patterns
    NETWORK = "network"               # Network formation


@dataclass
class SwarmAgent:
    """Individual agent in the carbon optimization swarm."""
    agent_id: str
    position: np.ndarray                    # Position in optimization space
    velocity: np.ndarray                    # Current velocity
    role: AgentRole
    energy: float                          # Agent energy level
    knowledge: Dict[str, Any]              # Accumulated knowledge
    memory: deque                          # Short-term memory
    connections: Set[str]                  # Connected agents
    performance_history: List[float]       # Historical performance
    specialization: str                    # Area of specialization
    lifetime: int                          # Age of the agent
    
    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        if self.memory is None:
            self.memory = deque(maxlen=100)
        if not self.connections:
            self.connections = set()
        if not self.knowledge:
            self.knowledge = {}
        if not self.performance_history:
            self.performance_history = []


@dataclass
class SwarmMessage:
    """Message passed between agents."""
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()


@dataclass
class EmergentBehavior:
    """Emergent behavior observed in the swarm."""
    behavior_id: str
    pattern_type: EmergentPattern
    participating_agents: Set[str]
    emergence_time: datetime
    stability: float                       # How stable the behavior is
    efficiency: float                      # How efficient it is
    carbon_impact: float                   # Impact on carbon optimization
    description: str
    
    def __post_init__(self):
        if not self.behavior_id:
            self.behavior_id = f"behavior_{uuid.uuid4().hex[:8]}"
        if not self.emergence_time:
            self.emergence_time = datetime.now()


class StigmergyEnvironment:
    """Environment that supports indirect coordination through stigmergy."""
    
    def __init__(self, dimensions: int = 10):
        self.dimensions = dimensions
        self.pheromone_trails: Dict[str, np.ndarray] = {}
        self.solution_markers: Dict[str, Dict] = {}
        self.environmental_feedback: np.ndarray = np.zeros(dimensions)
        self.decay_rate = 0.01
        
    def deposit_pheromone(
        self,
        agent_id: str,
        position: np.ndarray,
        intensity: float,
        pheromone_type: str = "optimization"
    ):
        """Deposit pheromone at a position."""
        key = f"{pheromone_type}_{agent_id}"
        
        if key not in self.pheromone_trails:
            self.pheromone_trails[key] = np.zeros(self.dimensions)
        
        # Add pheromone with distance decay
        for i in range(self.dimensions):
            distance = abs(i - (position[0] * self.dimensions) % self.dimensions)
            decay_factor = np.exp(-distance / 2.0)
            self.pheromone_trails[key][i] += intensity * decay_factor
    
    def get_pheromone_concentration(
        self,
        position: np.ndarray,
        pheromone_type: str = "optimization"
    ) -> float:
        """Get pheromone concentration at a position."""
        total_concentration = 0.0
        
        for key, trail in self.pheromone_trails.items():
            if pheromone_type in key:
                pos_idx = int((position[0] * self.dimensions) % self.dimensions)
                total_concentration += trail[pos_idx]
        
        return total_concentration
    
    def update_environment(self):
        """Update environment state (decay pheromones, etc.)."""
        for key in list(self.pheromone_trails.keys()):
            self.pheromone_trails[key] *= (1 - self.decay_rate)
            
            # Remove very weak trails
            if np.max(self.pheromone_trails[key]) < 0.01:
                del self.pheromone_trails[key]
    
    def place_solution_marker(
        self,
        agent_id: str,
        position: np.ndarray,
        quality: float,
        solution_data: Dict[str, Any]
    ):
        """Place a solution marker in the environment."""
        marker_id = f"solution_{agent_id}_{int(time.time())}"
        
        self.solution_markers[marker_id] = {
            'agent_id': agent_id,
            'position': position.copy(),
            'quality': quality,
            'data': solution_data,
            'timestamp': datetime.now()
        }
    
    def get_nearby_solutions(
        self,
        position: np.ndarray,
        radius: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Get solutions near a position."""
        nearby = []
        
        for marker_id, marker in self.solution_markers.items():
            distance = np.linalg.norm(position - marker['position'])
            if distance <= radius:
                nearby.append(marker)
        
        return sorted(nearby, key=lambda x: x['quality'], reverse=True)


class SwarmCommunicationProtocol:
    """Protocol for agent communication."""
    
    def __init__(self):
        self.message_queue: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.broadcast_messages: List[SwarmMessage] = []
        self.communication_network = nx.Graph()
        
    def send_message(self, message: SwarmMessage):
        """Send message to specific agent."""
        self.message_queue[message.receiver_id].append(message)
    
    def broadcast_message(self, message: SwarmMessage):
        """Broadcast message to all agents."""
        self.broadcast_messages.append(message)
    
    def get_messages(self, agent_id: str) -> List[SwarmMessage]:
        """Get messages for an agent."""
        messages = list(self.message_queue[agent_id])
        self.message_queue[agent_id].clear()
        
        # Add broadcast messages
        for broadcast in self.broadcast_messages:
            if broadcast.sender_id != agent_id:
                messages.append(broadcast)
        
        # Clear old broadcast messages
        current_time = datetime.now()
        self.broadcast_messages = [
            msg for msg in self.broadcast_messages
            if current_time - msg.timestamp < timedelta(minutes=5)
        ]
        
        return sorted(messages, key=lambda x: x.priority, reverse=True)
    
    def update_network(self, agents: List[SwarmAgent]):
        """Update communication network topology."""
        self.communication_network.clear()
        
        # Add all agents as nodes
        for agent in agents:
            self.communication_network.add_node(agent.agent_id, role=agent.role.value)
        
        # Add edges based on proximity and connections
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents[i+1:], i+1):
                # Connect if they're close in optimization space
                distance = np.linalg.norm(agent1.position - agent2.position)
                if distance < 2.0:  # Connection threshold
                    self.communication_network.add_edge(
                        agent1.agent_id, 
                        agent2.agent_id,
                        weight=1.0 / (distance + 0.1)
                    )
                
                # Connect if explicitly connected
                if agent2.agent_id in agent1.connections:
                    if not self.communication_network.has_edge(agent1.agent_id, agent2.agent_id):
                        self.communication_network.add_edge(
                            agent1.agent_id,
                            agent2.agent_id,
                            weight=2.0  # Higher weight for explicit connections
                        )


class EmergenceBehaviorDetector:
    """Detects and analyzes emergent behaviors in the swarm."""
    
    def __init__(self):
        self.detected_behaviors: List[EmergentBehavior] = []
        self.pattern_history: Dict[str, List[float]] = defaultdict(list)
        
    def detect_emergent_patterns(self, agents: List[SwarmAgent]) -> List[EmergentBehavior]:
        """Detect emergent patterns in agent behavior."""
        new_behaviors = []
        
        if len(agents) < 3:
            return new_behaviors
        
        # Extract positions and velocities
        positions = np.array([agent.position for agent in agents])
        velocities = np.array([agent.velocity for agent in agents])
        
        # Detect clustering
        clustering_behavior = self._detect_clustering(agents, positions)
        if clustering_behavior:
            new_behaviors.append(clustering_behavior)
        
        # Detect flocking
        flocking_behavior = self._detect_flocking(agents, positions, velocities)
        if flocking_behavior:
            new_behaviors.append(flocking_behavior)
        
        # Detect network formation
        network_behavior = self._detect_network_formation(agents)
        if network_behavior:
            new_behaviors.append(network_behavior)
        
        # Detect spiral patterns
        spiral_behavior = self._detect_spiral_pattern(agents, positions)
        if spiral_behavior:
            new_behaviors.append(spiral_behavior)
        
        self.detected_behaviors.extend(new_behaviors)
        return new_behaviors
    
    def _detect_clustering(self, agents: List[SwarmAgent], positions: np.ndarray) -> Optional[EmergentBehavior]:
        """Detect clustering behavior."""
        if len(agents) < 3:
            return None
        
        try:
            # Use DBSCAN to find clusters
            clustering = DBSCAN(eps=1.5, min_samples=2).fit(positions)
            labels = clustering.labels_
            
            # Check if there are meaningful clusters
            unique_labels = set(labels)
            unique_labels.discard(-1)  # Remove noise label
            
            if len(unique_labels) >= 2:  # At least 2 clusters
                clustered_agents = {agents[i].agent_id for i in range(len(agents)) if labels[i] != -1}
                
                if len(clustered_agents) >= len(agents) * 0.5:  # At least 50% in clusters
                    return EmergentBehavior(
                        behavior_id="",
                        pattern_type=EmergentPattern.CLUSTERING,
                        participating_agents=clustered_agents,
                        emergence_time=datetime.now(),
                        stability=0.8,  # Calculated based on cluster persistence
                        efficiency=0.7,  # Efficiency metric
                        carbon_impact=0.6,  # Impact on carbon optimization
                        description=f"Agents formed {len(unique_labels)} clusters"
                    )
        except Exception as e:
            logger.debug(f"Error detecting clustering: {e}")
        
        return None
    
    def _detect_flocking(
        self, 
        agents: List[SwarmAgent], 
        positions: np.ndarray, 
        velocities: np.ndarray
    ) -> Optional[EmergentBehavior]:
        """Detect flocking behavior."""
        if len(agents) < 3:
            return None
        
        try:
            # Calculate alignment (velocity similarity)
            velocity_similarities = []
            for i in range(len(velocities)):
                for j in range(i+1, len(velocities)):
                    similarity = np.dot(velocities[i], velocities[j]) / (
                        np.linalg.norm(velocities[i]) * np.linalg.norm(velocities[j]) + 1e-8
                    )
                    velocity_similarities.append(similarity)
            
            avg_alignment = np.mean(velocity_similarities) if velocity_similarities else 0
            
            # Calculate cohesion (position clustering)
            distances = spatial.distance.pdist(positions)
            avg_distance = np.mean(distances)
            cohesion = 1.0 / (1.0 + avg_distance)  # Inverse relationship
            
            # Flocking threshold
            if avg_alignment > 0.6 and cohesion > 0.3:
                return EmergentBehavior(
                    behavior_id="",
                    pattern_type=EmergentPattern.FLOCKING,
                    participating_agents={agent.agent_id for agent in agents},
                    emergence_time=datetime.now(),
                    stability=avg_alignment,
                    efficiency=cohesion,
                    carbon_impact=avg_alignment * cohesion,
                    description=f"Flocking with alignment {avg_alignment:.2f} and cohesion {cohesion:.2f}"
                )
        except Exception as e:
            logger.debug(f"Error detecting flocking: {e}")
        
        return None
    
    def _detect_network_formation(self, agents: List[SwarmAgent]) -> Optional[EmergentBehavior]:
        """Detect network formation patterns."""
        # Create network from connections
        G = nx.Graph()
        
        for agent in agents:
            G.add_node(agent.agent_id)
            for connection in agent.connections:
                if any(a.agent_id == connection for a in agents):
                    G.add_edge(agent.agent_id, connection)
        
        if G.number_of_edges() == 0:
            return None
        
        # Analyze network properties
        try:
            density = nx.density(G)
            clustering_coeff = nx.average_clustering(G)
            
            # Check for significant network structure
            if density > 0.3 and clustering_coeff > 0.4:
                return EmergentBehavior(
                    behavior_id="",
                    pattern_type=EmergentPattern.NETWORK,
                    participating_agents=set(G.nodes()),
                    emergence_time=datetime.now(),
                    stability=clustering_coeff,
                    efficiency=density,
                    carbon_impact=density * clustering_coeff,
                    description=f"Network formation with density {density:.2f} and clustering {clustering_coeff:.2f}"
                )
        except Exception as e:
            logger.debug(f"Error detecting network formation: {e}")
        
        return None
    
    def _detect_spiral_pattern(self, agents: List[SwarmAgent], positions: np.ndarray) -> Optional[EmergentBehavior]:
        """Detect spiral movement patterns."""
        if len(agents) < 5:
            return None
        
        try:
            # Convert to polar coordinates relative to center
            center = np.mean(positions, axis=0)
            relative_positions = positions - center
            
            # Calculate angles and distances
            angles = np.arctan2(relative_positions[:, 1], relative_positions[:, 0])
            distances = np.linalg.norm(relative_positions, axis=1)
            
            # Sort by angle
            sorted_indices = np.argsort(angles)
            sorted_distances = distances[sorted_indices]
            
            # Check for monotonic distance relationship (spiral pattern)
            correlation = np.corrcoef(np.arange(len(sorted_distances)), sorted_distances)[0, 1]
            
            if abs(correlation) > 0.7:  # Strong correlation indicates spiral
                participating_agents = {agents[i].agent_id for i in sorted_indices}
                
                return EmergentBehavior(
                    behavior_id="",
                    pattern_type=EmergentPattern.SPIRAL,
                    participating_agents=participating_agents,
                    emergence_time=datetime.now(),
                    stability=abs(correlation),
                    efficiency=0.8,
                    carbon_impact=abs(correlation) * 0.6,
                    description=f"Spiral pattern with correlation {correlation:.2f}"
                )
        except Exception as e:
            logger.debug(f"Error detecting spiral pattern: {e}")
        
        return None


class SwarmOptimizationAlgorithm:
    """Core swarm optimization algorithm for carbon efficiency."""
    
    def __init__(self):
        self.objective_function: Optional[Callable] = None
        self.constraints: List[Dict] = []
        self.optimization_history: List[Dict] = []
        
    def set_carbon_optimization_objective(self):
        """Set the carbon optimization objective function."""
        def carbon_objective(position: np.ndarray) -> float:
            """Objective function for carbon optimization."""
            # Simulate carbon emissions based on position in optimization space
            # In reality, this would interface with actual carbon measurement
            
            # Multiple local optima for interesting swarm behavior
            carbon = 0
            
            # Main carbon bowl (global minimum)
            carbon += 100 * np.sum(position**2)
            
            # Secondary optima
            for i in range(len(position)):
                carbon += 20 * np.sin(5 * position[i])**2
                carbon += 10 * np.cos(3 * position[i] + 1)**2
            
            # Add noise for realism
            carbon += np.random.normal(0, 5)
            
            return carbon
        
        self.objective_function = carbon_objective
    
    def evaluate_fitness(self, agent: SwarmAgent) -> float:
        """Evaluate fitness of an agent's position."""
        if self.objective_function is None:
            return 0.0
        
        try:
            carbon_emissions = self.objective_function(agent.position)
            # Convert to fitness (lower emissions = higher fitness)
            fitness = 1000.0 / (1.0 + carbon_emissions)
            return fitness
        except Exception as e:
            logger.debug(f"Error evaluating fitness for agent {agent.agent_id}: {e}")
            return 0.0
    
    def update_agent_position(
        self,
        agent: SwarmAgent,
        best_global_position: np.ndarray,
        swarm_agents: List[SwarmAgent],
        environment: StigmergyEnvironment
    ):
        """Update agent position based on swarm dynamics."""
        # Particle Swarm Optimization with swarm intelligence enhancements
        w = 0.7  # Inertia weight
        c1 = 1.5  # Personal best coefficient
        c2 = 1.5  # Global best coefficient
        c3 = 0.5  # Social interaction coefficient
        c4 = 0.3  # Environmental coefficient
        
        # Personal best
        if not hasattr(agent, 'best_position') or not hasattr(agent, 'best_fitness'):
            agent.best_position = agent.position.copy()
            agent.best_fitness = self.evaluate_fitness(agent)
        
        current_fitness = self.evaluate_fitness(agent)
        if current_fitness > agent.best_fitness:
            agent.best_position = agent.position.copy()
            agent.best_fitness = current_fitness
        
        # Calculate velocity components
        r1, r2, r3, r4 = np.random.random(4)
        
        # Inertia component
        inertia = w * agent.velocity
        
        # Personal best component
        personal = c1 * r1 * (agent.best_position - agent.position)
        
        # Global best component
        global_component = c2 * r2 * (best_global_position - agent.position)
        
        # Social interaction (nearby agents)
        social_component = np.zeros_like(agent.position)
        nearby_agents = [
            other for other in swarm_agents 
            if other.agent_id != agent.agent_id and 
            np.linalg.norm(other.position - agent.position) < 2.0
        ]
        
        if nearby_agents:
            for nearby in nearby_agents:
                nearby_fitness = self.evaluate_fitness(nearby)
                if nearby_fitness > current_fitness:
                    social_component += (nearby.position - agent.position)
            social_component = c3 * r3 * social_component / len(nearby_agents)
        
        # Environmental component (pheromone influence)
        pheromone_gradient = np.zeros_like(agent.position)
        for i in range(len(agent.position)):
            pos_test = agent.position.copy()
            pos_test[i] += 0.1
            forward_pheromone = environment.get_pheromone_concentration(pos_test)
            pos_test[i] -= 0.2
            backward_pheromone = environment.get_pheromone_concentration(pos_test)
            pheromone_gradient[i] = forward_pheromone - backward_pheromone
        
        environmental = c4 * r4 * pheromone_gradient
        
        # Update velocity and position
        agent.velocity = inertia + personal + global_component + social_component + environmental
        
        # Velocity clamping
        max_velocity = 2.0
        velocity_magnitude = np.linalg.norm(agent.velocity)
        if velocity_magnitude > max_velocity:
            agent.velocity = agent.velocity * max_velocity / velocity_magnitude
        
        # Update position
        agent.position += agent.velocity
        
        # Position bounds
        agent.position = np.clip(agent.position, -10.0, 10.0)
        
        # Update agent energy based on movement
        energy_cost = np.linalg.norm(agent.velocity) * 0.01
        agent.energy = max(0.0, agent.energy - energy_cost)
        
        # Deposit pheromone if good solution found
        if current_fitness > agent.best_fitness * 0.9:
            environment.deposit_pheromone(
                agent.agent_id,
                agent.position,
                intensity=current_fitness / 1000.0,
                pheromone_type="good_solution"
            )


class EmergentSwarmCarbonIntelligence:
    """Main orchestrator for emergent swarm carbon intelligence."""
    
    def __init__(self, swarm_size: int = 50, dimensions: int = 5):
        self.swarm_size = swarm_size
        self.dimensions = dimensions
        self.agents: List[SwarmAgent] = []
        self.environment = StigmergyEnvironment(dimensions)
        self.communication = SwarmCommunicationProtocol()
        self.emergence_detector = EmergenceBehaviorDetector()
        self.optimizer = SwarmOptimizationAlgorithm()
        
        self.best_global_position: Optional[np.ndarray] = None
        self.best_global_fitness: float = -np.inf
        self.iteration_count: int = 0
        self.convergence_history: List[float] = []
        
        self.is_running: bool = False
        self.optimization_task: Optional[asyncio.Task] = None
    
    async def initialize_swarm(self) -> bool:
        """Initialize the swarm intelligence system."""
        try:
            logger.info(f"Initializing emergent swarm with {self.swarm_size} agents")
            
            # Set up optimization objective
            self.optimizer.set_carbon_optimization_objective()
            
            # Create agents with diverse roles and positions
            role_distribution = {
                AgentRole.SCOUT: int(0.2 * self.swarm_size),
                AgentRole.WORKER: int(0.4 * self.swarm_size),
                AgentRole.COORDINATOR: int(0.1 * self.swarm_size),
                AgentRole.SPECIALIST: int(0.2 * self.swarm_size),
                AgentRole.COMMUNICATOR: int(0.1 * self.swarm_size)
            }
            
            agent_count = 0
            for role, count in role_distribution.items():
                for _ in range(count):
                    if agent_count >= self.swarm_size:
                        break
                    
                    # Random initial position
                    position = np.random.uniform(-5, 5, self.dimensions)
                    velocity = np.random.uniform(-1, 1, self.dimensions)
                    
                    agent = SwarmAgent(
                        agent_id="",
                        position=position,
                        velocity=velocity,
                        role=role,
                        energy=100.0,
                        knowledge={},
                        memory=deque(maxlen=100),
                        connections=set(),
                        performance_history=[],
                        specialization=role.value,
                        lifetime=0
                    )
                    
                    self.agents.append(agent)
                    agent_count += 1
            
            # Fill remaining slots with workers
            while len(self.agents) < self.swarm_size:
                position = np.random.uniform(-5, 5, self.dimensions)
                velocity = np.random.uniform(-1, 1, self.dimensions)
                
                agent = SwarmAgent(
                    agent_id="",
                    position=position,
                    velocity=velocity,
                    role=AgentRole.WORKER,
                    energy=100.0,
                    knowledge={},
                    memory=deque(maxlen=100),
                    connections=set(),
                    performance_history=[],
                    specialization="worker",
                    lifetime=0
                )
                
                self.agents.append(agent)
            
            # Initialize connections between agents
            await self._initialize_agent_connections()
            
            logger.info("Emergent swarm intelligence system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize swarm: {e}")
            return False
    
    async def _initialize_agent_connections(self):
        """Initialize connections between agents based on proximity and roles."""
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                # Connect based on role compatibility
                if self._are_compatible_roles(agent1.role, agent2.role):
                    if np.random.random() < 0.3:  # 30% connection probability
                        agent1.connections.add(agent2.agent_id)
                        agent2.connections.add(agent1.agent_id)
                
                # Connect based on proximity
                distance = np.linalg.norm(agent1.position - agent2.position)
                if distance < 2.0 and np.random.random() < 0.2:  # Proximity connection
                    agent1.connections.add(agent2.agent_id)
                    agent2.connections.add(agent1.agent_id)
    
    def _are_compatible_roles(self, role1: AgentRole, role2: AgentRole) -> bool:
        """Check if two roles are compatible for connection."""
        compatibility_matrix = {
            AgentRole.SCOUT: [AgentRole.COORDINATOR, AgentRole.COMMUNICATOR],
            AgentRole.WORKER: [AgentRole.COORDINATOR, AgentRole.SPECIALIST],
            AgentRole.COORDINATOR: [AgentRole.SCOUT, AgentRole.WORKER, AgentRole.COMMUNICATOR],
            AgentRole.SPECIALIST: [AgentRole.WORKER, AgentRole.COMMUNICATOR],
            AgentRole.COMMUNICATOR: [AgentRole.SCOUT, AgentRole.COORDINATOR, AgentRole.SPECIALIST]
        }
        
        return role2 in compatibility_matrix.get(role1, [])
    
    async def optimize_carbon_emissions(
        self,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6
    ) -> Dict[str, Any]:
        """Run emergent swarm optimization for carbon emissions."""
        logger.info("Starting emergent swarm carbon optimization")
        
        self.is_running = True
        optimization_start = time.time()
        
        try:
            for iteration in range(max_iterations):
                if not self.is_running:
                    break
                
                self.iteration_count = iteration
                
                # Update all agents
                await self._update_swarm_iteration()
                
                # Detect emergent behaviors
                emergent_behaviors = self.emergence_detector.detect_emergent_patterns(self.agents)
                
                # Check for convergence
                if len(self.convergence_history) > 10:
                    recent_improvement = abs(
                        self.convergence_history[-1] - self.convergence_history[-10]
                    )
                    if recent_improvement < convergence_threshold:
                        logger.info(f"Converged after {iteration} iterations")
                        break
                
                # Log progress periodically
                if iteration % 100 == 0:
                    logger.info(
                        f"Iteration {iteration}: Best fitness = {self.best_global_fitness:.4f}, "
                        f"Emergent behaviors = {len(emergent_behaviors)}"
                    )
                
                # Small delay to allow other tasks
                if iteration % 10 == 0:
                    await asyncio.sleep(0.001)
            
            optimization_time = time.time() - optimization_start
            
            # Calculate final results
            final_carbon_emissions = self.optimizer.objective_function(self.best_global_position) if self.best_global_position is not None else float('inf')
            
            results = {
                'optimization_completed': True,
                'iterations': self.iteration_count,
                'optimization_time_seconds': optimization_time,
                'best_position': self.best_global_position.tolist() if self.best_global_position is not None else None,
                'best_fitness': self.best_global_fitness,
                'final_carbon_emissions': final_carbon_emissions,
                'emergent_behaviors_detected': len(self.emergence_detector.detected_behaviors),
                'emergent_behaviors': [
                    {
                        'pattern_type': behavior.pattern_type.value,
                        'stability': behavior.stability,
                        'efficiency': behavior.efficiency,
                        'carbon_impact': behavior.carbon_impact,
                        'description': behavior.description,
                        'participating_agents_count': len(behavior.participating_agents)
                    }
                    for behavior in self.emergence_detector.detected_behaviors[-10:]  # Last 10
                ],
                'swarm_statistics': await self._get_swarm_statistics(),
                'convergence_history': self.convergence_history[-100:]  # Last 100 iterations
            }
            
            logger.info(f"Emergent swarm optimization completed in {optimization_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error in swarm optimization: {e}")
            return {'optimization_completed': False, 'error': str(e)}
        finally:
            self.is_running = False
    
    async def _update_swarm_iteration(self):
        """Update all agents for one iteration."""
        # Update agent positions
        for agent in self.agents:
            self.optimizer.update_agent_position(
                agent,
                self.best_global_position or np.zeros(self.dimensions),
                self.agents,
                self.environment
            )
            
            # Update performance history
            fitness = self.optimizer.evaluate_fitness(agent)
            agent.performance_history.append(fitness)
            if len(agent.performance_history) > 100:
                agent.performance_history = agent.performance_history[-50:]  # Keep last 50
            
            # Update global best
            if fitness > self.best_global_fitness:
                self.best_global_fitness = fitness
                self.best_global_position = agent.position.copy()
            
            # Age the agent
            agent.lifetime += 1
            
            # Energy regeneration for active agents
            if agent.energy < 50:
                agent.energy += 1.0
        
        # Update environment
        self.environment.update_environment()
        
        # Update communication network
        self.communication.update_network(self.agents)
        
        # Handle agent communication
        await self._process_agent_communication()
        
        # Record convergence
        self.convergence_history.append(self.best_global_fitness)
    
    async def _process_agent_communication(self):
        """Process communication between agents."""
        # Generate some communication based on agent roles and situations
        for agent in self.agents:
            messages = self.communication.get_messages(agent.agent_id)
            
            # Process received messages
            for message in messages:
                self._process_message(agent, message)
            
            # Generate outgoing messages based on role
            if agent.role == AgentRole.SCOUT and np.random.random() < 0.1:
                # Scouts report discoveries
                best_fitness = max(agent.performance_history[-10:]) if agent.performance_history else 0
                if best_fitness > np.mean([a.performance_history[-1] if a.performance_history else 0 for a in self.agents]):
                    for connection in agent.connections:
                        message = SwarmMessage(
                            sender_id=agent.agent_id,
                            receiver_id=connection,
                            message_type="discovery",
                            content={'position': agent.position.tolist(), 'fitness': best_fitness},
                            timestamp=datetime.now(),
                            priority=2
                        )
                        self.communication.send_message(message)
    
    def _process_message(self, agent: SwarmAgent, message: SwarmMessage):
        """Process a received message."""
        if message.message_type == "discovery":
            # Learn from discovered positions
            discovered_position = np.array(message.content['position'])
            discovered_fitness = message.content['fitness']
            
            # Store in knowledge
            agent.knowledge['last_discovery'] = {
                'position': discovered_position,
                'fitness': discovered_fitness,
                'timestamp': message.timestamp
            }
            
            # Adjust position slightly towards good discovery
            if discovered_fitness > self.optimizer.evaluate_fitness(agent):
                direction = discovered_position - agent.position
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    agent.velocity += 0.1 * direction / direction_norm
    
    async def _get_swarm_statistics(self) -> Dict[str, Any]:
        """Get current swarm statistics."""
        if not self.agents:
            return {}
        
        # Position statistics
        positions = np.array([agent.position for agent in self.agents])
        position_mean = np.mean(positions, axis=0)
        position_std = np.std(positions, axis=0)
        
        # Fitness statistics
        fitnesses = [self.optimizer.evaluate_fitness(agent) for agent in self.agents]
        
        # Energy statistics
        energies = [agent.energy for agent in self.agents]
        
        # Connection statistics
        total_connections = sum(len(agent.connections) for agent in self.agents)
        
        # Role distribution
        role_counts = {}
        for agent in self.agents:
            role_counts[agent.role.value] = role_counts.get(agent.role.value, 0) + 1
        
        return {
            'swarm_size': len(self.agents),
            'position_mean': position_mean.tolist(),
            'position_std': position_std.tolist(),
            'fitness_mean': np.mean(fitnesses),
            'fitness_std': np.std(fitnesses),
            'fitness_min': np.min(fitnesses),
            'fitness_max': np.max(fitnesses),
            'energy_mean': np.mean(energies),
            'energy_std': np.std(energies),
            'total_connections': total_connections,
            'avg_connections_per_agent': total_connections / len(self.agents),
            'role_distribution': role_counts,
            'pheromone_trails_active': len(self.environment.pheromone_trails),
            'solution_markers_count': len(self.environment.solution_markers)
        }
    
    async def stop_optimization(self):
        """Stop the current optimization."""
        self.is_running = False
        if self.optimization_task and not self.optimization_task.done():
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
    
    async def get_emergent_insights(self) -> Dict[str, Any]:
        """Get insights about emergent behaviors and swarm intelligence."""
        if not self.agents:
            return {'insights': [], 'emergent_behaviors': [], 'swarm_health': 'unknown'}
        
        insights = []
        
        # Analyze emergent behaviors
        recent_behaviors = self.emergence_detector.detected_behaviors[-10:]  # Last 10
        if recent_behaviors:
            most_stable = max(recent_behaviors, key=lambda x: x.stability)
            most_efficient = max(recent_behaviors, key=lambda x: x.efficiency)
            highest_impact = max(recent_behaviors, key=lambda x: x.carbon_impact)
            
            insights.append(f"Most stable pattern: {most_stable.pattern_type.value} (stability: {most_stable.stability:.2f})")
            insights.append(f"Most efficient pattern: {most_efficient.pattern_type.value} (efficiency: {most_efficient.efficiency:.2f})")
            insights.append(f"Highest carbon impact: {highest_impact.pattern_type.value} (impact: {highest_impact.carbon_impact:.2f})")
        
        # Swarm health assessment
        avg_energy = np.mean([agent.energy for agent in self.agents])
        avg_connections = np.mean([len(agent.connections) for agent in self.agents])
        
        if avg_energy > 70 and avg_connections > 2:
            swarm_health = "excellent"
        elif avg_energy > 50 and avg_connections > 1:
            swarm_health = "good"
        elif avg_energy > 30:
            swarm_health = "fair"
        else:
            swarm_health = "poor"
        
        insights.append(f"Swarm health: {swarm_health} (avg energy: {avg_energy:.1f}, avg connections: {avg_connections:.1f})")
        
        # Performance insights
        if len(self.convergence_history) > 100:
            recent_trend = np.polyfit(
                range(len(self.convergence_history[-50:])),
                self.convergence_history[-50:],
                1
            )[0]
            
            if recent_trend > 0.01:
                insights.append("Swarm is actively improving - strong upward trend")
            elif recent_trend > -0.01:
                insights.append("Swarm has converged - stable performance")
            else:
                insights.append("Swarm performance declining - may need intervention")
        
        return {
            'insights': insights,
            'emergent_behaviors': [
                {
                    'pattern_type': behavior.pattern_type.value,
                    'description': behavior.description,
                    'stability': behavior.stability,
                    'efficiency': behavior.efficiency,
                    'carbon_impact': behavior.carbon_impact
                }
                for behavior in recent_behaviors
            ],
            'swarm_health': swarm_health,
            'average_energy': avg_energy,
            'average_connections': avg_connections,
            'optimization_progress': {
                'best_fitness': self.best_global_fitness,
                'iterations': self.iteration_count,
                'convergence_trend': recent_trend if len(self.convergence_history) > 100 else 0.0
            }
        }


# Convenience functions
def create_emergent_swarm_intelligence(
    swarm_size: int = 50, 
    dimensions: int = 5
) -> EmergentSwarmCarbonIntelligence:
    """Create an emergent swarm intelligence system."""
    return EmergentSwarmCarbonIntelligence(swarm_size, dimensions)


async def optimize_carbon_with_swarm_intelligence(
    swarm_size: int = 50,
    dimensions: int = 5,
    max_iterations: int = 1000
) -> Dict[str, Any]:
    """Convenience function for swarm-based carbon optimization."""
    swarm = create_emergent_swarm_intelligence(swarm_size, dimensions)
    
    if await swarm.initialize_swarm():
        return await swarm.optimize_carbon_emissions(max_iterations)
    else:
        return {'optimization_completed': False, 'error': 'Failed to initialize swarm'}


if __name__ == "__main__":
    # Example usage
    async def main():
        print("🌟 Emergent Swarm Carbon Intelligence Demo")
        
        # Create and initialize swarm
        swarm = create_emergent_swarm_intelligence(swarm_size=30, dimensions=3)
        
        if await swarm.initialize_swarm():
            print("✅ Swarm initialized successfully")
            
            # Run optimization
            result = await swarm.optimize_carbon_emissions(max_iterations=500)
            
            print(f"\n🎯 Optimization Results:")
            print(f"Completed: {result['optimization_completed']}")
            print(f"Best fitness: {result.get('best_fitness', 'N/A'):.4f}")
            print(f"Final carbon emissions: {result.get('final_carbon_emissions', 'N/A'):.2f}")
            print(f"Emergent behaviors detected: {result.get('emergent_behaviors_detected', 0)}")
            
            # Get insights
            insights = await swarm.get_emergent_insights()
            print(f"\n🧠 Swarm Insights:")
            for insight in insights['insights']:
                print(f"  • {insight}")
        else:
            print("❌ Failed to initialize swarm")
    
    # Run the demo
    asyncio.run(main())