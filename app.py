import json
import os
import uuid
import time
import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import argparse
from dotenv import load_dotenv

# Scientific computing and NLP
from scipy.spatial.distance import cosine, euclidean, cityblock, minkowski
from openai import AsyncOpenAI

# Load .env variables
load_dotenv()

# Create async OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def get_embeddings(all_texts):
    # Create embeddings for a list of strings
    response = await client.embeddings.create(
        model="text-embedding-3-large", 
        input=all_texts
    )
    return [item.embedding for item in response.data]

#Threshold value
THRESHOLD = 0.1

# Asynchronous operations in a synchronous Dash context
import nest_asyncio
nest_asyncio.apply()

def clean_for_logging(text: str) -> str:
    """Remove or replace Unicode characters that can't be encoded in cp1252 for logging."""
    import re
    # Replace common problematic Unicode characters
    text = re.sub(r'[🗺️🏨⭐📋🤖🔍✅❌🛠️🔄]', '[EMOJI]', text)
    # Remove any remaining non-ASCII characters that might cause issues
    text = text.encode('ascii', errors='ignore').decode('ascii')
    return text

try:
    from agent.agent import TravelAgent, get_agent_for_tool, AGENT_REGISTRY
except ImportError:
    print("Warning: Agent module not found. Please ensure 'agent.py' is in the agent directory.")
    TravelAgent = None

# Import pydantic_ai message types for proper parsing
try:
    from pydantic_ai.messages import (
        ModelMessage, ModelRequest, ModelResponse, 
        ToolCallPart, ToolReturnPart, TextPart, 
        SystemPromptPart, UserPromptPart
    )
    from pydantic_ai import RunContext, Agent
except ImportError:
    print("Warning: pydantic_ai messages not found.")
    ModelMessage = None

# --- Logging Setup ---
APP_LOGS = []
class ListLogHandler(logging.Handler):
    def emit(self, record):
        APP_LOGS.insert(0, self.format(record))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Console and memory handler
    log_handler = ListLogHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    
    # File handler for debugging
    file_handler = logging.FileHandler('travel_agent_debug.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# --- Core Data Structures and Classes ---
class SimilarityMethod(Enum):
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    MINKOWSKI = "minkowski"
    JACCARD = "jaccard"

class InfluenceType(Enum):
    DIRECT = "direct"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    LLM_DERIVED = "llm_derived"

@dataclass
class ToolInfluence:
    source_tool_idx: int
    target_tool_idx: int
    influence_type: InfluenceType
    influence_score: float
    influence_explanation: str

@dataclass
class PreComputedEmbeddings:
    input_embedding: np.ndarray
    output_embedding: np.ndarray
    input_rope_embedding: np.ndarray
    output_rope_embedding: np.ndarray
    timestamp: float

@dataclass
class ToolCall:
    tool_name: str
    tool_input: Any
    tool_output: Any
    timestamp: float
    duration: float
    tool_idx: int = -1
    tool_call_id: Optional[str] = None
    influences_from: List[Dict] = field(default_factory=list)
    embeddings: Optional[PreComputedEmbeddings] = None
    has_meaningful_input: bool = True  # New field to track if tool has meaningful input
    # Multi-agent tracking fields
    agent_name: Optional[str] = None  # Which agent executed this tool
    agent_model: Optional[str] = None  # Which model was used
    is_delegation: bool = False  # Whether this is a delegation call
    delegated_to: Optional[str] = None  # Which agent this was delegated to

@dataclass
class InfluenceChain:
    tool_idx: int
    tool_name: str
    influence_score: float
    influence_type: str
    explanation: str
    level: int
    connected_tools: List[int] = field(default_factory=list)

@dataclass
class GroupedToolExecution:
    """Represents a complete tool execution with call and return grouped together"""
    tool_number: int
    tool_name: str
    tool_input: Any
    tool_output: Any
    has_call: bool
    has_return: bool
    tool_call_id: Optional[str] = None
    call_index: Optional[int] = None
    return_index: Optional[int] = None
    agent_name: Optional[str] = None
    agent_model: Optional[str] = None

class AdvancedRoPEEnhancer:
    def __init__(self, d_model: int = 3072):
        self.d_model = d_model

    def apply_rope(self, vector: np.ndarray, position: float, temperature: float = 10000.0) -> np.ndarray:
        if not isinstance(vector, np.ndarray) or vector.size == 0: 
            return np.array([])
        original_size = vector.shape[0]
        if original_size % 2 != 0: 
            vector = np.append(vector, 0)
        d_model = vector.shape[0]
        theta = position / (temperature ** (np.arange(0, d_model, 2, dtype=np.float32) / d_model))
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        v_pairs = vector.reshape(-1, 2)
        rotated_pairs = np.zeros_like(v_pairs)
        rotated_pairs[:, 0] = v_pairs[:, 0] * cos_theta - v_pairs[:, 1] * sin_theta
        rotated_pairs[:, 1] = v_pairs[:, 0] * sin_theta + v_pairs[:, 1] * cos_theta
        return rotated_pairs.flatten()[:original_size]

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray, method: SimilarityMethod) -> float:
        if vec1.size == 0 or vec2.size == 0 or vec1.shape != vec2.shape: 
            return 0.0
        try:
            if method == SimilarityMethod.COSINE: 
                return max(0.0, 1 - cosine(vec1, vec2))
            if method == SimilarityMethod.DOT_PRODUCT: 
                return max(0.0, np.dot(vec1/np.linalg.norm(vec1), vec2/np.linalg.norm(vec2)))
            if method == SimilarityMethod.EUCLIDEAN: 
                return 1 / (1 + euclidean(vec1, vec2))
            if method == SimilarityMethod.MANHATTAN: 
                return 1 / (1 + cityblock(vec1, vec2))
            if method == SimilarityMethod.MINKOWSKI: 
                return 1 / (1 + minkowski(vec1, vec2, p=3))
            if method == SimilarityMethod.JACCARD:
                bin1=(vec1>np.percentile(vec1,75))
                bin2=(vec2>np.percentile(vec2,75))
                return np.sum(bin1&bin2)/np.sum(bin1|bin2) if np.sum(bin1|bin2)>0 else 0.0
        except Exception: 
            return 0.0
        return 0.0

    def get_combined_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        methods = [
            SimilarityMethod.COSINE,
            SimilarityMethod.DOT_PRODUCT
  

        ]
        scores = [self.compute_similarity(vec1, vec2, method) for method in methods]
        return sum(scores) / len(scores)  # Take average to keep values between 0-1

class AdvancedCausalAnalyzer:
    def __init__(self):
        self.rope_enhancer = AdvancedRoPEEnhancer(d_model=3072)

    async def create_enhanced_session(self, response_text: str, tool_calls: List[ToolCall], query: str) -> Dict:
        """Create session with pre-computed RoPE embeddings for all tools"""
        logger.info(f"Creating enhanced session with {len(tool_calls)} tool calls")
        
        # Filter out final planner agent from causal analysis
        analysis_tool_calls = [tc for tc in tool_calls if tc.agent_name != "agent_planner"]
        logger.info(f"Filtered out final planner agent: analyzing {len(analysis_tool_calls)} tools for causal analysis")
        
        # Add tool indices and check for meaningful inputs
        for idx, tc in enumerate(tool_calls):
            tc.tool_idx = idx
            tc.has_meaningful_input = self._has_meaningful_input(tc.tool_input)
            # Debug logging for each tool
            logger.info(f"Tool {idx+1} ({tc.tool_name}): input={clean_for_logging(str(tc.tool_input))}, has_meaningful_input={tc.has_meaningful_input}")
            
            # Special handling for known parameter-less tools
            if tc.tool_name in ['get_current_location', 'get_current_date_time'] or tc.tool_input == {}:
                tc.has_meaningful_input = False
                logger.info(f"Forced parameter-less for {tc.tool_name}: has_meaningful_input=False")
        
        # Pre-compute embeddings and RoPE transformations for all tools (including planner for completeness)
        await self._precompute_all_embeddings(tool_calls, query, response_text)
        
        # Calculate comprehensive influences using O(n²) approach focusing on output-to-output (exclude planner)
        self._calculate_output_to_output_influences(analysis_tool_calls)
        
        # Calculate cascaded causal analysis for multi-agent interactions (exclude planner)
        self._calculate_agent_to_agent_influences(analysis_tool_calls)
        self._calculate_cascaded_tool_influences(analysis_tool_calls)
        
        logger.info("Session created with pre-computed embeddings and O(n²) output-to-output analysis")
        
        return {
            "tool_calls": [self._serialize_tool_call(tc) for tc in tool_calls],
            "response_text": response_text,
            "query": query,
            "embeddings_computed": True,
            "total_tools": len(tool_calls)
        }

    async def _precompute_all_embeddings(self, tool_calls: List[ToolCall], query: str, response_text: str):
        """Pre-compute embeddings and RoPE transformations for all tool inputs and outputs"""
        logger.info("Pre-computing embeddings for all tools...")
        
        # Prepare all texts for batch embedding
        all_texts = []
        text_indices = {}  # Map text position to (tool_idx, 'input'/'output')
        tools_with_input = 0
        
        for idx, tc in enumerate(tool_calls):
            tool_input_str = to_string_for_pre(tc.tool_input)
            tool_output_str = to_string_for_pre(tc.tool_output)
            
            # Debug log the input detection
            logger.info(f"Tool {tc.tool_name}: input={clean_for_logging(str(tc.tool_input))}, has_meaningful_input={tc.has_meaningful_input}")
            
            # Only create input embeddings for tools with actual input parameters
            if tc.has_meaningful_input:
                input_text = f"Tool: {tc.tool_name} | Query Context: {query[:200]} | Input: {tool_input_str[:800]}"
                text_indices[len(all_texts)] = (idx, 'input')
                all_texts.append(input_text)
                tools_with_input += 1
                logger.info(f"Creating input embedding for {tc.tool_name}")
            else:
                logger.info(f"Skipping input embedding for parameter-less tool: {tc.tool_name}")
            
            # Always create output embeddings
            output_text = f"Tool: {tc.tool_name} | Query Context: {query[:200]} | Output: {tool_output_str[:800]}"
            text_indices[len(all_texts)] = (idx, 'output')
            all_texts.append(output_text)
        
        logger.info(f"Processing {len(tool_calls)} tools: {tools_with_input} with meaningful input, {len(tool_calls) - tools_with_input} parameter-less")
        
        # Also add the final response for LLM detection
        response_text_formatted = f"Final Response: {response_text[:1000]}"
        all_texts.append(response_text_formatted)
        response_idx = len(all_texts) - 1
        
        # Batch embed all texts
        try:
            all_embeddings = await get_embeddings(all_texts)
            logger.info(f"Successfully embedded {len(all_embeddings)} texts")
            
            # Store response embedding for LLM detection
            response_embedding = np.array(all_embeddings[response_idx])
            response_rope = self.rope_enhancer.apply_rope(response_embedding, time.time())
            
            # Apply RoPE transformations and store embeddings
            for text_idx, embedding in enumerate(all_embeddings[:-1]):  # Exclude response embedding
                tool_idx, io_type = text_indices[text_idx]
                tc = tool_calls[tool_idx]
                
                if tc.embeddings is None:
                    tc.embeddings = PreComputedEmbeddings(
                        input_embedding=np.array([]),
                        output_embedding=np.array([]),
                        input_rope_embedding=np.array([]),
                        output_rope_embedding=np.array([]),
                        timestamp=tc.timestamp
                    )
                
                base_embedding = np.array(embedding)
                rope_embedding = self.rope_enhancer.apply_rope(base_embedding, tc.timestamp)
                
                if io_type == 'input':
                    # Only store input embeddings for tools with meaningful input
                    if tc.has_meaningful_input:
                        tc.embeddings.input_embedding = base_embedding
                        tc.embeddings.input_rope_embedding = rope_embedding
                        logger.info(f"Stored input embeddings for {tc.tool_name}")
                    else:
                        logger.warning(f"Attempted to store input embedding for parameter-less tool: {tc.tool_name}")
                else:  # output
                    tc.embeddings.output_embedding = base_embedding
                    tc.embeddings.output_rope_embedding = rope_embedding
                    
                    # Calculate LLM derivation score
                    llm_similarity = self.rope_enhancer.get_combined_similarity(
                        rope_embedding, response_rope
                    )
                    tc.llm_derivation_score = llm_similarity
                    logger.info(f"Stored output embeddings for {tc.tool_name}")
                    
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")

    def _calculate_agent_to_agent_influences(self, tool_calls: List[ToolCall]):
        """Calculate O(n²) agent-to-agent influences for multi-agent causal analysis"""
        logger.info("Calculating agent-to-agent influences...")
        
        # Group tools by agent
        agent_groups = {}
        for tool_call in tool_calls:
            agent = tool_call.agent_name or "master_coordinator"
            if agent not in agent_groups:
                agent_groups[agent] = []
            agent_groups[agent].append(tool_call)
        
        # Calculate influences between agents
        for target_agent, target_tools in agent_groups.items():
            for source_agent, source_tools in agent_groups.items():
                if source_agent == target_agent:
                    continue
                    
                # Find the strongest influence between agents
                max_influence = 0.0
                best_source_tool = None
                best_target_tool = None
                
                for source_tool in source_tools:
                    for target_tool in target_tools:
                        if source_tool.timestamp >= target_tool.timestamp:
                            continue
                            
                        # Calculate cross-agent influence
                        if (source_tool.embeddings and target_tool.embeddings and 
                            source_tool.embeddings.output_rope_embedding.size > 0 and 
                            target_tool.embeddings.input_rope_embedding.size > 0):
                            
                            influence = self.rope_enhancer.get_combined_similarity(
                                source_tool.embeddings.output_rope_embedding,
                                target_tool.embeddings.input_rope_embedding
                            )
                            
                            if influence > max_influence:
                                max_influence = influence
                                best_source_tool = source_tool
                                best_target_tool = target_tool
                
                # Add significant agent-to-agent influences
                if max_influence > 0.3 and best_source_tool and best_target_tool:
                    best_target_tool.influences_from.append({
                        "source_tool_idx": best_source_tool.tool_idx,
                        "influence_type": "agent_to_agent",
                        "influence_score": max_influence,
                        "explanation": f"Agent {source_agent} influenced agent {target_agent} (cross-agent communication)"
                    })
        
        logger.info("Completed agent-to-agent influence analysis")

    def _calculate_cascaded_tool_influences(self, tool_calls: List[ToolCall]):
        """Calculate cascaded tool-to-tool influences within and across agents"""
        logger.info("Calculating cascaded tool-to-tool influences...")
        
        n_tools = len(tool_calls)
        
        # Create cascaded influence matrix
        for i in range(n_tools):
            for j in range(i + 1, n_tools):  # Only forward influences
                source_tool = tool_calls[i]
                target_tool = tool_calls[j]
                
                # Calculate direct tool-to-tool influence
                direct_influence = 0.0
                if (source_tool.embeddings and target_tool.embeddings and 
                    source_tool.embeddings.output_rope_embedding.size > 0 and 
                    target_tool.embeddings.input_rope_embedding.size > 0):
                    
                    direct_influence = self.rope_enhancer.get_combined_similarity(
                        source_tool.embeddings.output_rope_embedding,
                        target_tool.embeddings.input_rope_embedding
                    )
                
                # Calculate indirect (cascaded) influences through intermediate tools
                cascaded_influence = 0.0
                for k in range(i + 1, j):
                    intermediate_tool = tool_calls[k]
                    
                    if (source_tool.embeddings and intermediate_tool.embeddings and
                        intermediate_tool.embeddings and target_tool.embeddings and
                        source_tool.embeddings.output_rope_embedding.size > 0 and
                        intermediate_tool.embeddings.input_rope_embedding.size > 0 and
                        intermediate_tool.embeddings.output_rope_embedding.size > 0 and
                        target_tool.embeddings.input_rope_embedding.size > 0):
                        
                        # Calculate path influence: source -> intermediate -> target
                        path_influence_1 = self.rope_enhancer.get_combined_similarity(
                            source_tool.embeddings.output_rope_embedding,
                            intermediate_tool.embeddings.input_rope_embedding
                        )
                        path_influence_2 = self.rope_enhancer.get_combined_similarity(
                            intermediate_tool.embeddings.output_rope_embedding,
                            target_tool.embeddings.input_rope_embedding
                        )
                        
                        # Multiply influences for cascaded effect
                        path_influence = path_influence_1 * path_influence_2
                        cascaded_influence = max(cascaded_influence, path_influence)
                
                # Combine direct and cascaded influences
                total_influence = max(direct_influence, cascaded_influence * 0.8)  # Slight penalty for indirect
                
                if total_influence > 0.25:
                    influence_type = "cascaded_tool" if cascaded_influence > direct_influence else "direct_tool"
                    
                    target_tool.influences_from.append({
                        "source_tool_idx": source_tool.tool_idx,
                        "influence_type": influence_type,
                        "influence_score": total_influence,
                        "explanation": f"Tool cascaded influence: {source_tool.tool_name} -> {target_tool.tool_name} " +
                                     f"({'direct' if influence_type == 'direct_tool' else 'through intermediate tools'})"
                    })
        
        logger.info("Completed cascaded tool-to-tool influence analysis")

    def _has_meaningful_input(self, tool_input: Any) -> bool:
        """Check if a tool has meaningful input parameters"""
        if tool_input is None:
            return False
        
        if isinstance(tool_input, dict):
            # Empty dict means no parameters (only RunContext)
            if not tool_input:
                return False
            # Check if all values are empty/None/meaningless
            meaningful_values = []
            for k, v in tool_input.items():
                # Skip RunContext-related keys and empty values
                if (v is not None and v != "" and v != {} and v != [] and 
                    not str(k).lower().startswith('ctx') and 
                    not str(v).lower().startswith('runcontext')):
                    meaningful_values.append(v)
            return len(meaningful_values) > 0
        
        if isinstance(tool_input, (list, tuple)):
            return len(tool_input) > 0 and any(item is not None for item in tool_input)
        
        if isinstance(tool_input, str):
            return tool_input.strip() != ""
        
        # For other types, consider them meaningful if not None
        return tool_input is not None

    def _calculate_output_to_output_influences(self, tool_calls: List[ToolCall]):
        """Calculate influences using O(n²) output-to-output embedding similarity"""
        logger.info("Calculating O(n²) output-to-output influences...")
        
        n_tools = len(tool_calls)
        
        # Create similarity matrix for all tool outputs
        similarity_matrix = np.zeros((n_tools, n_tools))
        
        for i in range(n_tools):
            for j in range(n_tools):
                if i == j:
                    continue
                    
                tool_i, tool_j = tool_calls[i], tool_calls[j]
                
                if (tool_i.embeddings and tool_j.embeddings and 
                    tool_i.embeddings.output_rope_embedding.size > 0 and 
                    tool_j.embeddings.output_rope_embedding.size > 0):
                    
                    similarity = self.rope_enhancer.get_combined_similarity(
                        tool_i.embeddings.output_rope_embedding,
                        tool_j.embeddings.output_rope_embedding
                    )
                    similarity_matrix[i][j] = similarity
        
        # Build influences based on similarity and temporal ordering
        for target_idx in range(n_tools):
            target_tool = tool_calls[target_idx]
            
            for source_idx in range(n_tools):
                if source_idx >= target_idx:  # Only consider earlier tools
                    continue
                    
                source_tool = tool_calls[source_idx]
                output_similarity = similarity_matrix[source_idx][target_idx]
                
                # Enhanced similarity threshold (no artificial cap)
                if output_similarity > 0.2:
                    # Calculate temporal influence
                    temporal_score = self._calculate_enhanced_temporal_influence(source_tool, target_tool)
                    
                    # Combined influence score
                    combined_score =(output_similarity +  temporal_score)/2
                    
                    if combined_score > 0.25:
                        influence_type = self._determine_influence_type(
                            source_tool, target_tool, output_similarity, temporal_score
                        )
                        
                        explanation = self._generate_influence_explanation(
                            source_tool, target_tool, influence_type, combined_score, 
                            output_similarity, temporal_score
                        )
                        
                        target_tool.influences_from.append({
                            'source_tool_idx': source_idx,
                            'target_tool_idx': target_idx,
                            'influence_type': influence_type.value,
                            'influence_score': combined_score,
                            'output_similarity': output_similarity,
                            'temporal_score': temporal_score,
                            'influence_explanation': explanation
                        })
        
        # Log influence statistics
        total_influences = sum(len(tc.influences_from) for tc in tool_calls)
        logger.info(f"Calculated {total_influences} total output-to-output influences across {n_tools} tools")

    def _determine_influence_type(self, source: ToolCall, target: ToolCall, 
                                 output_sim: float, temporal_score: float) -> InfluenceType:
        """Determine the type of influence based on various factors"""
        # Check for LLM derivation
        if hasattr(target, 'llm_derivation_score') and target.llm_derivation_score > 0.6:
            return InfluenceType.LLM_DERIVED
        
        # High output similarity suggests semantic relationship
        if output_sim > 0.7:
            return InfluenceType.SEMANTIC
        
        # High temporal score with moderate similarity suggests direct flow
        if temporal_score > 0.6 and output_sim > 0.4:
            return InfluenceType.DIRECT
        
        return InfluenceType.SEMANTIC

    def _generate_influence_explanation(self, source: ToolCall, target: ToolCall, 
                                       influence_type: InfluenceType, combined_score: float,
                                       output_sim: float, temporal_score: float) -> str:
        """Generate detailed explanation for the influence"""
        base_explanation = f"{source.tool_name} → {target.tool_name}"
        
        if influence_type == InfluenceType.LLM_DERIVED:
            return f"{base_explanation}: Output content derived by LLM from tool results (similarity: {output_sim:.3f})"
        elif influence_type == InfluenceType.DIRECT:
            return f"{base_explanation}: Direct data flow detected (output similarity: {output_sim:.3f}, temporal: {temporal_score:.3f})"
        elif influence_type == InfluenceType.SEMANTIC:
            return f"{base_explanation}: Semantic relationship in outputs (similarity: {output_sim:.3f})"
        else:
            return f"{base_explanation}: Temporal influence (score: {combined_score:.3f})"

    def _calculate_enhanced_temporal_influence(self, source: ToolCall, target: ToolCall) -> float:
        """Enhanced temporal influence calculation"""
        if target.timestamp <= source.timestamp:
            return 0.0
        
        time_diff = target.timestamp - source.timestamp
        
        # Very close temporal proximity (likely sequential)
        if time_diff < 1.0:
            return 0.9
        elif time_diff < 3.0:
            return 0.8
        elif time_diff < 10.0:
            return max(0.4, 0.8 - (time_diff / 25.0))
        elif time_diff < 30.0:
            return max(0.2, 0.5 - (time_diff / 60.0))
        
        return 0.1

    async def analyze_selection_with_precomputed(self, selected_text: str, session_data: Dict) -> Dict:
        """Analyze selection using pre-computed embeddings with improved scoring"""
        if not session_data.get('embeddings_computed'):
            return {"error": "No pre-computed embeddings available"}
        
        tool_calls = [self._deserialize_tool_call(tc_data) for tc_data in session_data['tool_calls']]
        query = session_data['query']
        
        logger.info(f"Analyzing selection: '{selected_text[:50]}...' using pre-computed embeddings")
        
        # Find best matching tools using pre-computed embeddings (no artificial caps)
        best_matches = await self._find_best_matches_enhanced(selected_text, tool_calls, query)
        
        if not best_matches:
            return {"error": "No tool matches found, it is generated by LLM non-dependant on tool from its trained params", "selected_text": selected_text}
        
        # Build enhanced influence chains
        influence_chains = self._build_output_based_influence_chains(best_matches, tool_calls)
        
        # Detect LLM-generated vs tool-derived content
        content_analysis = self._analyze_content_origin(selected_text, tool_calls, query)
        
        return {
            "selected_text": selected_text,
            "best_matches": best_matches,
            "influence_chains": influence_chains,
            "content_analysis": content_analysis,
            "analysis_type": "enhanced_precomputed_embeddings"
        }

    async def _find_best_matches_enhanced(self, selected_text: str, 
                                        tool_calls: List[ToolCall], query: str) -> List[Dict]:
        """Find best tool matches with enhanced scoring focused on output for parameter-less tools"""
        
        # Embed the selected text
        cleaned_selection = self._clean_text_for_analysis(selected_text)
        selection_text = f"Query: {query} | Selected text: {cleaned_selection}"
        
        try:
            selection_embeddings = await get_embeddings([selection_text])
            selection_embedding = np.array(selection_embeddings[0])
            selection_rope = self.rope_enhancer.apply_rope(selection_embedding, time.time())
        except Exception as e:
            logger.error(f"Error embedding selection: {e}")
            return []
        
        tool_matches = []
        
        for tc in tool_calls:
            if not tc.embeddings:
                continue
            
            # Skip final planner agent tools from enhanced analysis matching
            if tc.agent_name == "agent_planner":
                continue
            
            # Calculate similarities based on whether tool has meaningful input
            if tc.has_meaningful_input and tc.embeddings.input_embedding.size > 0:
                # Tool has meaningful input - consider both input and output
                input_rope_sim = self.rope_enhancer.get_combined_similarity(
                    selection_rope, tc.embeddings.input_rope_embedding
                )
                input_base_sim = self.rope_enhancer.get_combined_similarity(
                    selection_embedding, tc.embeddings.input_embedding
                )
                
                # Always calculate output similarities
                output_rope_sim = self.rope_enhancer.get_combined_similarity(
                    selection_rope, tc.embeddings.output_rope_embedding
                )
                output_base_sim = self.rope_enhancer.get_combined_similarity(
                    selection_embedding, tc.embeddings.output_embedding
                )
                
                # For tools with meaningful input, consider both input and output
                rope_score = max(input_rope_sim, output_rope_sim)
                base_score = max(input_base_sim, output_base_sim)
                
            else:
                # Tool has no meaningful input (like get_current_location) - focus only on output
                input_rope_sim = 0.0
                input_base_sim = 0.0
                
                output_rope_sim = self.rope_enhancer.get_combined_similarity(
                    selection_rope, tc.embeddings.output_rope_embedding
                )
                output_base_sim = self.rope_enhancer.get_combined_similarity(
                    selection_embedding, tc.embeddings.output_embedding
                )
                
                # For parameter-less tools, only consider output
                rope_score = output_rope_sim
                base_score = output_base_sim
                
                logger.info(f"Parameter-less tool {tc.tool_name}: input_rope_sim={input_rope_sim}, input_base_sim={input_base_sim}")
            
            # Weighted combination with emphasis on RoPE
            combined_score = (rope_score + base_score)/2
            
            # Text-based overlap detection (focused on output)
            tool_output_str = to_string_for_pre(tc.tool_output)
            text_overlap_score = self._calculate_text_overlap_score(cleaned_selection, tool_output_str)
            
            # Final score - take the maximum to avoid suppressing high similarities
            final_score = (combined_score + text_overlap_score) / 2
            
            # Apply threshold
            if final_score > THRESHOLD:
                tool_number = tc.tool_idx + 1
                
                # Force input similarities to exactly 0.0 for parameter-less tools
                if not tc.has_meaningful_input:
                    input_rope_sim = 0.0
                    input_base_sim = 0.0
                
                tool_matches.append({
                    "tool_idx": tc.tool_idx,
                    "tool_number": tool_number,
                    "tool_name": tc.tool_name,
                    "agent_name": tc.agent_name,
                    "agent_model": tc.agent_model,
                    "is_delegation": tc.is_delegation,
                    "delegated_to": tc.delegated_to,
                    "similarity_score": final_score,
                    "rope_input_sim": input_rope_sim,
                    "rope_output_sim": output_rope_sim,
                    "base_input_sim": input_base_sim,
                    "base_output_sim": output_base_sim,
                    "text_overlap_score": text_overlap_score,
                    "tool_output": tool_output_str,
                    "tool_input": to_string_for_pre(tc.tool_input),
                    "has_meaningful_input": tc.has_meaningful_input,
                    "debug_trace_ref": f"Tool #{tool_number}: {tc.tool_name}"
                })
        
        # Sort and return matches
        tool_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        logger.info(f"Found {len(tool_matches)} enhanced tool matches")
        
        return tool_matches[:5]  # Return top 5 matches

    def _calculate_text_overlap_score(self, selection: str, tool_output: str) -> float:
        """Enhanced text overlap calculation"""
        selection_clean = re.sub(r'\W+', ' ', selection.lower()).strip()
        output_clean = re.sub(r'\W+', ' ', tool_output.lower()).strip()
        
        if not selection_clean or not output_clean:
            return 0.0
        
        # Exact substring match (high score)
        if len(selection_clean) > 10 and selection_clean in output_clean:
            return 0.95
        
        # Phrase matching
        selection_words = selection_clean.split()
        phrase_matches = 0
        total_phrases = 0
        
        if len(selection_words) >= 3:
            for i in range(len(selection_words) - 2):
                phrase = " ".join(selection_words[i:i+3])
                total_phrases += 1
                if phrase in output_clean:
                    phrase_matches += 1
        
        if total_phrases > 0:
            phrase_score = phrase_matches / total_phrases
            if phrase_score > 0.5:
                return min(0.9, phrase_score)
        
        # Word overlap
        sel_words = set(selection_words)
        out_words = set(output_clean.split())
        overlap = len(sel_words.intersection(out_words))
        
        if len(sel_words) > 0:
            overlap_ratio = overlap / len(sel_words)
            if overlap_ratio > 0.6:
                return min(0.8, overlap_ratio)
        
        return 0.0

    def _build_output_based_influence_chains(self, best_matches: List[Dict], 
                                           tool_calls: List[ToolCall]) -> List[Dict]:
        """Build influence chains based on output-to-output similarities"""
        logger.info(f"Building output-based influence chains for {len(best_matches)} matches")
        
        influence_chains = []
        
        for match in best_matches:
            tool_idx = match['tool_idx']
            chain = self._build_output_chain_recursively(tool_idx, tool_calls, level=0, max_depth=4)
            
            if chain:
                chain_depth = self._count_chain_depth(chain)
                
                influence_chains.append({
                    "root_tool": match,
                    "chain": self._serialize_influence_chain(chain),
                    "chain_length": chain_depth,
                    "chain_type": "output_based"
                })
        
        logger.info(f"Built {len(influence_chains)} output-based influence chains")
        return influence_chains

    def _build_output_chain_recursively(self, tool_idx: int, tool_calls: List[ToolCall], 
                                      level: int, max_depth: int, visited: set = None) -> Optional[InfluenceChain]:
        """Build chain focusing on output-to-output relationships"""
        if visited is None:
            visited = set()
        
        if level >= max_depth or tool_idx >= len(tool_calls) or tool_idx in visited:
            return None
        
        visited.add(tool_idx)
        current_tool = tool_calls[tool_idx]
        
        # Create chain node
        chain_node = InfluenceChain(
            tool_idx=tool_idx,
            tool_name=current_tool.tool_name,
            influence_score=1.0 if level == 0 else max(0.2, 0.85 ** level),
            influence_type="root" if level == 0 else "output_derived",
            explanation=self._get_output_level_explanation(current_tool, level),
            level=level
        )
        
        # Find strongest output-based influence
        strongest_influence = None
        max_output_sim = 0.0
        
        for influence in current_tool.influences_from:
            output_sim = influence.get('output_similarity', 0.0)
            if output_sim > max_output_sim:
                max_output_sim = output_sim
                strongest_influence = influence
        
        if strongest_influence and level < max_depth - 1:
            parent_tool_idx = strongest_influence['source_tool_idx']
            
            if parent_tool_idx not in visited and parent_tool_idx < len(tool_calls):
                parent_chain = self._build_output_chain_recursively(
                    parent_tool_idx, tool_calls, level + 1, max_depth, visited.copy()
                )
                
                if parent_chain:
                    chain_node.connected_tools = [parent_tool_idx]
                    parent_tool_name = tool_calls[parent_tool_idx].tool_name
                    chain_node.explanation = f"Level {level}: Output influenced by {parent_tool_name} (similarity: {max_output_sim:.3f})"
                    setattr(chain_node, 'parent_chain', parent_chain)
        
        return chain_node

    def _analyze_content_origin(self, selected_text: str, tool_calls: List[ToolCall], query: str) -> Dict:
        """Analyze whether content is LLM-generated or tool-derived"""
        content_analysis = {
            "is_llm_generated": False,
            "tool_derived_confidence": 0.0,
            "llm_generated_confidence": 0.0,
            "source_analysis": []
        }
        
        # Check for high tool-derived confidence
        max_tool_confidence = 0.0
        for tc in tool_calls:
            if hasattr(tc, 'llm_derivation_score'):
                tool_confidence = 1.0 - tc.llm_derivation_score  # Inverse for tool derivation
                max_tool_confidence = max(max_tool_confidence, tool_confidence)
                
                content_analysis["source_analysis"].append({
                    "tool_name": tc.tool_name,
                    "tool_idx": tc.tool_idx,
                    "tool_derivation_confidence": tool_confidence,
                    "llm_derivation_confidence": tc.llm_derivation_score
                })
        
        content_analysis["tool_derived_confidence"] = max_tool_confidence
        content_analysis["llm_generated_confidence"] = 1.0 - max_tool_confidence
        content_analysis["is_llm_generated"] = max_tool_confidence < 0.6
        
        return content_analysis

    def _get_output_level_explanation(self, tool: ToolCall, level: int) -> str:
        """Generate explanation focused on output relationships"""
        if level == 0:
            return f"Primary match: {tool.tool_name} output directly matches selected content"
        elif level == 1:
            return f"Output influence: {tool.tool_name} output semantically influenced the primary match"
        else:
            return f"Level {level} output chain: {tool.tool_name} contributed to the semantic chain"

    # Utility methods remain the same...
    def _count_chain_depth(self, chain: Optional[InfluenceChain]) -> int:
        if not chain:
            return 0
        depth = 1
        current = chain
        while hasattr(current, 'parent_chain') and current.parent_chain:
            depth += 1
            current = current.parent_chain
        return depth

    def _serialize_influence_chain(self, chain: Optional[InfluenceChain]) -> Optional[Dict]:
        if not chain:
            return None
        
        result = {
            "tool_idx": chain.tool_idx,
            "tool_name": chain.tool_name,
            "influence_score": chain.influence_score,
            "influence_type": chain.influence_type,
            "explanation": chain.explanation,
            "level": chain.level,
            "connected_tools": chain.connected_tools
        }
        
        if hasattr(chain, 'parent_chain') and chain.parent_chain:
            result["parent_chain"] = self._serialize_influence_chain(chain.parent_chain)
        
        return result

    def _clean_text_for_analysis(self, text: str) -> str:
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _serialize_tool_call(self, tc: ToolCall) -> Dict:
        embeddings_data = None
        if tc.embeddings:
            embeddings_data = {
                "input_embedding": tc.embeddings.input_embedding.tolist() if tc.embeddings.input_embedding.size > 0 else [],
                "output_embedding": tc.embeddings.output_embedding.tolist() if tc.embeddings.output_embedding.size > 0 else [],
                "input_rope_embedding": tc.embeddings.input_rope_embedding.tolist() if tc.embeddings.input_rope_embedding.size > 0 else [],
                "output_rope_embedding": tc.embeddings.output_rope_embedding.tolist() if tc.embeddings.output_rope_embedding.size > 0 else [],
                "timestamp": tc.embeddings.timestamp
            }
        
        return {
            "tool_name": tc.tool_name,
            "tool_input": tc.tool_input,
            "tool_output": tc.tool_output,
            "timestamp": tc.timestamp,
            "duration": tc.duration,
            "tool_idx": tc.tool_idx,
            "tool_call_id": tc.tool_call_id,
            "influences_from": tc.influences_from,
            "embeddings": embeddings_data,
            "llm_derivation_score": getattr(tc, 'llm_derivation_score', 0.0),
            "has_meaningful_input": getattr(tc, 'has_meaningful_input', True),
            # Multi-agent tracking fields
            "agent_name": getattr(tc, 'agent_name', None),
            "agent_model": getattr(tc, 'agent_model', None),
            "is_delegation": getattr(tc, 'is_delegation', False),
            "delegated_to": getattr(tc, 'delegated_to', None)
        }

    def _deserialize_tool_call(self, tc_data: Dict) -> ToolCall:
        embeddings = None
        if tc_data.get("embeddings"):
            emb_data = tc_data["embeddings"]
            embeddings = PreComputedEmbeddings(
                input_embedding=np.array(emb_data["input_embedding"]) if emb_data["input_embedding"] else np.array([]),
                output_embedding=np.array(emb_data["output_embedding"]) if emb_data["output_embedding"] else np.array([]),
                input_rope_embedding=np.array(emb_data["input_rope_embedding"]) if emb_data["input_rope_embedding"] else np.array([]),
                output_rope_embedding=np.array(emb_data["output_rope_embedding"]) if emb_data["output_rope_embedding"] else np.array([]),
                timestamp=emb_data["timestamp"]
            )
        
        tc = ToolCall(
            tool_name=tc_data["tool_name"],
            tool_input=tc_data["tool_input"],
            tool_output=tc_data["tool_output"],
            timestamp=tc_data["timestamp"],
            duration=tc_data["duration"],
            tool_idx=tc_data.get("tool_idx", -1),
            tool_call_id=tc_data.get("tool_call_id"),
            influences_from=tc_data.get("influences_from", []),
            embeddings=embeddings
        )
        
        tc.llm_derivation_score = tc_data.get("llm_derivation_score", 0.0)
        tc.has_meaningful_input = tc_data.get("has_meaningful_input", True)
        
        # Multi-agent tracking fields
        tc.agent_name = tc_data.get("agent_name", None)
        tc.agent_model = tc_data.get("agent_model", None)
        tc.is_delegation = tc_data.get("is_delegation", False)
        tc.delegated_to = tc_data.get("delegated_to", None)
        
        # Re-validate has_meaningful_input in case serialization failed
        if tc.has_meaningful_input:
            actual_has_input = self._has_meaningful_input(tc.tool_input)
            if not actual_has_input:
                tc.has_meaningful_input = False
                logger.info(f"Corrected has_meaningful_input for {tc.tool_name}: was True, should be False")
        
        return tc

# Initialize the analyzer
analyzer = AdvancedCausalAnalyzer()

# --- Enhanced Dash App Layout with Draggable Sidebar ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True)

# Add custom CSS for resizable sidebar and search highlighting
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
        .resizer {
            width: 5px;
            background: #444;
            cursor: col-resize;
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            z-index: 1000;
        }
        .resizer:hover {
            background: #007bff;
        }
        .analysis-sidebar-container {
            position: relative;
            transition: width 0.3s ease;
        }
        .analysis-content {
            padding-left: 10px;
        }
        .search-highlight {
            background-color: #ffeb3b !important;
            color: #000 !important;
            padding: 1px 2px;
            border-radius: 2px;
        }
        .tool-content {
            line-height: 1.4;
        }
        .trace-item {
            transition: opacity 0.2s ease;
        }
        .trace-item.filtered-out {
            opacity: 0.3;
        }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    dcc.Store(id="session-store", data={}),
    dcc.Store(id="current-session-id"),
    dcc.Store(id="analysis-trigger-store"),
    dcc.Store(id="last-processed-timestamp-store", data=None),
    dcc.Store(id="sidebar-width-store", data=300),  # Store for analysis sidebar width in pixels
    dcc.Store(id="trace-search-store", data=""),  # Store for trace search filter
    dcc.Interval(id='selection-poller', interval=500),
    
    # Floating analyze button
    html.Div(id='floating-button-container', children=[
        dbc.Button("🔍 Analyze Selection", id='floating-analyze-btn', size='sm', color="info")
    ], style={'position': 'absolute', 'zIndex': 1000, 'display': 'none'}),
    
    # Header
    html.Header(className="text-center my-4", children=[
        html.H2("🧠 Enhanced Travel Agent with RoPE & Chain Analysis"),
        dbc.Badge("3072D Embeddings + Enhanced O(n²) Analysis + Parameter-less Tool Handling", color="success")
    ]),
    
    # Main layout with resizable sidebar
    dbc.Row([
        # Sessions sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("📚 Sessions", className="mb-0"),
                    dbc.Button("⟵", id="collapse-sidebar-btn", size="sm", color="secondary", className="ms-auto")
                ], className="d-flex align-items-center"), 
                dbc.CardBody(id="session-list", style={"height": "75vh", "overflowY": "auto"})
            ])
        ], id="sidebar-col", width=3, style={"transition": "all 0.3s ease"}),
        
        # Main content area
        dbc.Col([
            # Query input card
            dbc.Card(dbc.CardBody([
                dbc.Textarea(id="query-input", placeholder="Type your travel query...", rows=4, className="mb-2"),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(
                        id="model-dropdown", 
                        options=[
                            {"label": "OpenAI: gpt-4o", "value": "openai:gpt-4o"},
                            {"label": "OpenAI: gpt-5", "value": "gpt-5-2025-08-07"},
                            {"label": "Anthropic: claude-4-sonnet", "value": "anthropic:claude-sonnet-4-20250514"},
                            {"label": "Anthropic: claude-4-opus", "value": "claude-opus-4-20250514"}], 
                        value="openai:gpt-4o"
                    ), width=9),
                    dbc.Col(dbc.Button("Send", id="run-button", className="w-100", color="primary"), width=3),
                ]),
            ])),
            
            # Loading and main content with tabs
            dcc.Loading(id="loading-main", children=[
                html.Div(id="response-tabs-container", className="mt-3")
            ]),
        ], id="main-content-col", width="auto", style={"flex": "1"}),
        
        # Analysis sidebar (resizable)
        html.Div([
            # Resizer handle
            html.Div(className="resizer", id="analysis-resizer"),
            
            # Analysis content
            dbc.Collapse([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("⛓️ Enhanced Influence Analysis", className="mb-0"),
                        dbc.Badge("RoPE + O(n²) + Parameter-less Tool Handling", color="success", className="ms-2"),
                        dbc.Button("✕", id="close-analysis-btn", size="sm", color="outline-secondary", className="ms-auto")
                    ], className="d-flex align-items-center"),
                    dbc.CardBody(
                        dcc.Loading(html.Div(id="causal-analysis-content"), className="analysis-content"), 
                        style={"height": "75vh", "overflowY": "auto"}
                    )
                ])
            ], id="analysis-sidebar", is_open=False)
        ], id="analysis-sidebar-container", className="analysis-sidebar-container", 
           style={"width": "300px", "position": "relative"})
    ], className="flex-fill", style={"display": "flex"}),
    
    # Debug logs
    dbc.Row(dbc.Col(dbc.Accordion([
        dbc.AccordionItem(
            dcc.Loading(html.Pre(id='log-viewer', style={
                'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all', 'maxHeight': '300px', 'overflowY': 'auto'
            })), 
            title="📋 Debug Logs & System Status"
        )
    ], start_collapsed=True), className="mt-3"))
], fluid=True, className="vh-100 d-flex flex-column")

# --- Clientside Callbacks for UI Interactions ---
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks === 0 || n_clicks === null) {
            return dash_clientside.no_update;
        }
        const selection = window.getSelection();
        const selectionText = selection.toString().trim();
        if (selectionText) {
            return {text: selectionText, timestamp: Date.now()};
        }
        return dash_clientside.no_update;
    }
    """,
    Output("analysis-trigger-store", "data"),
    Input("floating-analyze-btn", "n_clicks"),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_intervals) {
        const selection = window.getSelection();
        const selectionText = selection.toString().trim();
        const targetNode = document.getElementById('final-response-text');
        
        if (selectionText && targetNode && selection.anchorNode && targetNode.contains(selection.anchorNode)) {
            const range = selection.getRangeAt(0);
            const rect = range.getBoundingClientRect();
            const top = rect.bottom + window.scrollY + 5;
            const left = rect.right + window.scrollX + 5;
            
            return {
                'position': 'absolute', 
                'zIndex': 1000, 
                'display': 'block', 
                'top': `${top}px`, 
                'left': `${left}px`
            };
        }
        return {'display': 'none'};
    }
    """,
    Output("floating-button-container", "style"),
    Input("selection-poller", "n_intervals"),
    prevent_initial_call=True
)

# Resizable sidebar functionality
app.clientside_callback(
    """
    function() {
        let isResizing = false;
        let startX = 0;
        let startWidth = 0;
        
        const resizer = document.getElementById('analysis-resizer');
        const sidebar = document.getElementById('analysis-sidebar-container');
        
        if (!resizer || !sidebar) {
            return dash_clientside.no_update;
        }
        
        resizer.addEventListener('mousedown', function(e) {
            isResizing = true;
            startX = e.clientX;
            startWidth = parseInt(window.getComputedStyle(sidebar).width, 10);
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', stopResize);
            e.preventDefault();
        });
        
        function handleMouseMove(e) {
            if (!isResizing) return;
            const width = startWidth - (e.clientX - startX);
            if (width >= 250 && width <= 600) {
                sidebar.style.width = width + 'px';
            }
        }
        
        function stopResize() {
            isResizing = false;
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', stopResize);
        }
        
        return dash_clientside.no_update;
    }
    """,
    Output("sidebar-width-store", "data", allow_duplicate=True),
    Input("analysis-sidebar", "is_open"),
    prevent_initial_call=True
)

# Sidebar collapse functionality
@app.callback(
    [Output("sidebar-col", "width"), Output("main-content-col", "style")],
    Input("collapse-sidebar-btn", "n_clicks"),
    State("sidebar-col", "width"),
    prevent_initial_call=True
)
def toggle_sidebar(n_clicks, current_width):
    if not n_clicks:
        raise PreventUpdate
    
    if current_width == 3:
        return 1, {"flex": "1", "marginLeft": "10px"}  # Collapsed
    else:
        return 3, {"flex": "1"}  # Expanded

# Close analysis sidebar
@app.callback(
    Output("analysis-sidebar", "is_open", allow_duplicate=True),
    Input("close-analysis-btn", "n_clicks"),
    prevent_initial_call=True
)
def close_analysis_sidebar(n_clicks):
    if n_clicks:
        return False
    raise PreventUpdate

# Store trace search filter
@app.callback(
    Output("trace-search-store", "data"),
    Input("trace-search-input", "value"),
    prevent_initial_call=True
)
def update_trace_search(search_value):
    return search_value or ""

# --- FIXED Main Server-side Callbacks ---
@app.callback(
    [Output("response-tabs-container", "children"), 
     Output("session-store", "data"), 
     Output("current-session-id", "data")],
    Input("run-button", "n_clicks"),
    [State("query-input", "value"), 
     State("model-dropdown", "value"), 
     State("session-store", "data")],
    prevent_initial_call=True
)
def run_agent_and_create_session(n_clicks, query, model, sessions):
    if not n_clicks or not query: 
        raise PreventUpdate
        
    logger.info(f"Running agent for query: '{query}'")
    sessions = sessions or {}
    
    try:
        travel_agent = TravelAgent()
        resp, trace = travel_agent.run(query, model=model)
        
        # Use proper pydantic_ai message parsing instead of custom trace format
        if isinstance(trace, bytes): 
            trace = json.loads(trace.decode('utf-8'))
        
        # Extract tool calls using proper pydantic_ai message structure
        logger.info(f"[DEBUG] Starting tool call extraction from trace with {len(trace) if trace else 0} messages")
        tool_calls = extract_tool_calls_from_messages(trace or [], model)
        logger.info(f"[DEBUG] Extracted {len(tool_calls)} tool calls")
        for i, tc in enumerate(tool_calls):
            logger.info(f"[DEBUG] ToolCall #{i}: {tc.tool_name} -> agent: {tc.agent_name}")
        
        # Create enhanced session with pre-computed embeddings
        analysis_results = asyncio.run(
            analyzer.create_enhanced_session(resp, tool_calls, query)
        )

        session_id = str(uuid.uuid4())[:8]
        sessions[session_id] = {
            "id": session_id, 
            "query": query, 
            "response": resp, 
            "model": model,
            "timestamp": datetime.now().isoformat(), 
            "analysis": analysis_results,
            "trace": ensure_json_serializable(trace)
        }
        
        logger.info(f"Enhanced session {session_id} created with {len(tool_calls)} tool calls")
        layout = create_tabbed_response_layout(resp, sessions[session_id]['trace'], query, tool_calls)
        
        return layout, sessions, session_id
        
    except Exception as e:
        logger.exception("Error running agent.")
        return dbc.Alert(f"Agent Error: {e}", color="danger"), sessions, no_update

# FIXED: Improved causal analysis callback with better state handling
@app.callback(
    [Output("causal-analysis-content", "children"), 
     Output("analysis-sidebar", "is_open", allow_duplicate=True)],
    Input("analysis-trigger-store", "data"),
    [State("current-session-id", "data"), 
     State("session-store", "data"),
     State("analysis-sidebar", "is_open")],
    prevent_initial_call=True
)
def display_enhanced_analysis(selection_data, session_id, sessions, sidebar_is_open):
    if not selection_data or not selection_data.get('text') or not session_id:
        raise PreventUpdate
    
    session = sessions.get(session_id, {})
    analysis_data = session.get('analysis', {})
    selected_text = selection_data['text']

    if not analysis_data or not analysis_data.get('embeddings_computed'):
        return dbc.Alert("No pre-computed embeddings for this session.", color="warning"), True
        
    logger.info(f"Enhanced analysis for: '{selected_text[:50]}...'")
    
    try:
        detailed_analysis = asyncio.run(
            analyzer.analyze_selection_with_precomputed(selected_text, analysis_data)
        )
        
        if "error" in detailed_analysis:
            error_msg = detailed_analysis["error"]
            return dbc.Alert(f"Analysis Error: {error_msg}", color="warning"), True
        
        analysis_layout = create_enhanced_analysis_layout(detailed_analysis)
        return analysis_layout, True
        
    except Exception as e:
        logger.exception("Error in enhanced analysis")
        return dbc.Alert(f"Analysis failed: {str(e)}", color="danger"), True

@app.callback(
    Output("session-list", "children"),
    Input("session-store", "data")
)
def update_session_list(sessions):
    if not sessions: 
        return [html.P("No sessions yet.", className="text-muted text-center p-3")]
    
    sorted_sessions = sorted(sessions.values(), key=lambda x: x['timestamp'], reverse=True)
    
    session_cards = []
    for s in sorted_sessions:
        try:
            total_tools = s.get('analysis', {}).get('total_tools', 0)
            embeddings_computed = s.get('analysis', {}).get('embeddings_computed', False)
            
            card = dbc.Card(dbc.CardBody([
                html.H6(s['query'][:40] + "...", className="card-title"),
                html.P([
                    f"ID: {s['id']} | Tools: {total_tools} | ",
                    dbc.Badge("✓", color="success" if embeddings_computed else "secondary", size="sm")
                ], className="card-text small text-muted")
            ]), 
            id={"type": "session-card", "index": s['id']}, 
            className="mb-2", 
            style={'cursor': 'pointer', 'transition': 'all 0.2s ease'})
            
            session_cards.append(card)
        except Exception as e:
            logger.error(f"Error creating session card: {e}")
            continue
    
    return session_cards if session_cards else [html.P("No valid sessions found.", className="text-muted text-center p-3")]

@app.callback(
    [Output("response-tabs-container", "children", allow_duplicate=True),
     Output("current-session-id", "data", allow_duplicate=True)],
    Input({"type": "session-card", "index": ALL}, "n_clicks"),
    State("session-store", "data"),
    prevent_initial_call=True
)
def load_session(n_clicks, sessions):
    ctx = callback_context
    if not ctx.triggered or not any(n_clicks): 
        raise PreventUpdate
        
    session_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']
    session = sessions.get(session_id)
    
    if not session: 
        return dbc.Alert("Session not found.", color="danger"), no_update
        
    logger.info(f"Loading session {session_id}")
    
    # Reconstruct tool_calls from session analysis data for consistent display
    session_tool_calls = None
    if session.get('analysis', {}).get('tool_calls'):
        session_tool_calls = [
            analyzer._deserialize_tool_call(tc_data) 
            for tc_data in session['analysis']['tool_calls']
        ]
    
    return create_tabbed_response_layout(
        session['response'], 
        session['trace'], 
        session['query'], 
        session_tool_calls
    ), session_id

@app.callback(
    Output("log-viewer", "children"), 
    Input("session-store", "data"),
    prevent_initial_call=True
)
def update_log_viewer(_): 
    return "\n".join(APP_LOGS[:50])  # Show last 50 log entries

# --- FIXED Tool Call Extraction Functions ---

def _extract_agent_info(tool_name: str, tool_input: Any, tool_output: str, model: str = "openai:gpt-4o") -> Tuple[Optional[str], Optional[str], bool, Optional[str]]:
    """
    Extract agent information from tool calls for multi-agent tracking using dynamic agent registry.
    
    Returns:
        Tuple of (agent_name, agent_model, is_delegation, delegated_to)
    """
    # Check if this is a delegation tool
    if tool_name.startswith('delegate_'):
        is_delegation = True
        agent_name = "master_coordinator"
        agent_model = model
        
        # Extract which agent was delegated to based on tool name
        delegation_map = {
            'delegate_research': "research_agent",
            'delegate_planning': "planning_agent", 
            'delegate_location_analysis': "location_agent",
            'delegate_timing_analysis': "timing_agent"
        }
        delegated_to = delegation_map.get(tool_name, "unknown_agent")
            
        return agent_name, agent_model, is_delegation, delegated_to
    
    # Use dynamic agent detection from registry
    try:
        agent_name, _ = get_agent_for_tool(tool_name)
        logger.info(f"[DEBUG] Tool '{tool_name}' assigned to agent: {agent_name} by dynamic registry")
        return agent_name, model, False, None
    except Exception as e:
        logger.info(f"[DEBUG] Dynamic agent detection failed for '{tool_name}': {e}")
    
    # Look for agent information in the tool output if it's a dict
    try:
        if isinstance(tool_output, str) and tool_output.startswith('{') and '"agent"' in tool_output:
            import json
            output_dict = json.loads(tool_output)
            if 'agent' in output_dict and 'model' in output_dict:
                return output_dict['agent'], output_dict['model'], False, None
    except:
        pass
    
    # Default to master coordinator for regular tools
    return "master_coordinator", model, False, None

def extract_tool_calls_from_messages(messages: List[Dict], model: str = "openai:gpt-4o") -> List[ToolCall]:
    """
    Extract tool calls from pydantic_ai message structure.
    Updated for latest pydantic_ai patterns with multi-agent tracking.
    """
    logger.info(f"Extracting tool calls from {len(messages)} messages using latest pydantic_ai structure")
    
    tool_calls = []
    call_buffer = {}  # Store tool calls by tool_call_id
    tool_call_counter = 0
    
    for msg_idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
            
        # Look for parts in the message
        parts = message.get('parts', [])
        if not parts:
            continue
            
        for part_idx, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
                
            part_kind = part.get('part_kind', '')
            
            if part_kind == 'tool-call':
                tool_call_counter += 1
                tool_name = part.get('tool_name', f'unknown_tool_{msg_idx}_{part_idx}')
                logger.info(f"[DEBUG] Found tool-call #{tool_call_counter}: {tool_name} in message {msg_idx}, part {part_idx}")
                tool_call_id = part.get('tool_call_id')
                
                # Use a combination approach for matching
                buffer_key = tool_call_id if tool_call_id else f"{tool_name}_{tool_call_counter}"
                
                # Store the tool call
                call_buffer[buffer_key] = {
                    'tool_name': tool_name,
                    'tool_input': part.get('args', {}),
                    'timestamp': time.time() + tool_call_counter * 0.1,
                    'tool_number': tool_call_counter,
                    'tool_call_id': tool_call_id
                }
                logger.info(f"Buffered tool call #{tool_call_counter}: {tool_name} (Key: {buffer_key})")
                
            elif part_kind == 'tool-return':
                tool_name = part.get('tool_name', f'unknown_tool_{msg_idx}_{part_idx}')
                tool_call_id = part.get('tool_call_id')
                logger.info(f"[DEBUG] Found tool-return: {tool_name} in message {msg_idx}, part {part_idx}")
                
                # Try multiple matching strategies
                matching_key = None
                call_data = None
                
                # Strategy 1: Exact tool_call_id match
                if tool_call_id and tool_call_id in call_buffer:
                    matching_key = tool_call_id
                    call_data = call_buffer[matching_key]
                
                # Strategy 2: Find by tool_name if no exact ID match
                if not call_data:
                    for key, buffered_call in call_buffer.items():
                        if buffered_call['tool_name'] == tool_name:
                            matching_key = key
                            call_data = buffered_call
                            break
                
                # Strategy 3: Use most recent call if multiple matches
                if not call_data and call_buffer:
                    # Get the most recent call (highest tool_number)
                    matching_key = max(call_buffer.keys(), 
                                     key=lambda k: call_buffer[k]['tool_number'])
                    call_data = call_buffer[matching_key]
                
                if call_data:
                    # Remove from buffer and create ToolCall
                    call_buffer.pop(matching_key)
                    
                    # Determine agent information from tool name and output
                    agent_name, agent_model, is_delegation, delegated_to = _extract_agent_info(
                        call_data['tool_name'], 
                        call_data['tool_input'], 
                        part.get('content', ''),
                        model
                    )
                    logger.info(f"[DEBUG] Tool '{call_data['tool_name']}' assigned to agent: {agent_name} by _extract_agent_info")
                    
                    tool_call = ToolCall(
                        tool_name=call_data['tool_name'],
                        tool_input=call_data['tool_input'],
                        tool_output=part.get('content', ''),
                        timestamp=call_data['timestamp'],
                        duration=0.1,
                        tool_idx=len(tool_calls),
                        tool_call_id=call_data.get('tool_call_id'),
                        agent_name=agent_name,
                        agent_model=agent_model,
                        is_delegation=is_delegation,
                        delegated_to=delegated_to
                    )
                    tool_calls.append(tool_call)
                    logger.info(f"[DEBUG] Completed tool call pair: {call_data['tool_name']} (Key: {matching_key})")
                    logger.info(f"[DEBUG] Final ToolCall object agent_name: {tool_call.agent_name}")
                else:
                    # Create orphaned return - this should be rare with proper pydantic_ai
                    tool_call = ToolCall(
                        tool_name=tool_name,
                        tool_input="[Orphaned return - no matching call found]",
                        tool_output=part.get('content', ''),
                        timestamp=time.time(),
                        duration=0.1,
                        tool_idx=len(tool_calls),
                        tool_call_id=tool_call_id
                    )
                    tool_calls.append(tool_call)
                    logger.warning(f"Created orphaned return for: {tool_name}")
    
    # Handle any remaining orphaned calls
    for buffer_key, call_data in call_buffer.items():
        tool_call = ToolCall(
            tool_name=call_data['tool_name'],
            tool_input=call_data['tool_input'],
            tool_output="[No return received - incomplete tool execution]",
            timestamp=call_data['timestamp'],
            duration=0.1,
            tool_idx=len(tool_calls),
            tool_call_id=call_data.get('tool_call_id')
        )
        tool_calls.append(tool_call)
        logger.warning(f"Created orphaned call for: {call_data['tool_name']}")
    
    logger.info(f"Successfully extracted {len(tool_calls)} tool calls with improved matching")
    return tool_calls

# --- Enhanced Layout Creation Functions ---

def create_tabbed_response_layout(response: str, trace: List[Dict], query: str, session_tool_calls: List[ToolCall] = None) -> dbc.Card:
    """Create tabbed layout separating response and traces with search functionality"""
    # Use the same tool calls that were used for analysis to ensure perfect consistency
    if session_tool_calls:
        tool_executions = convert_tool_calls_to_executions(session_tool_calls)
    else:
        # Fallback to parsing trace if tool calls not available
        tool_executions = extract_grouped_tool_executions(trace)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("🤖 Agent Response & Execution", className="mb-0"),
            dbc.Badge(f"{len(tool_executions)} agent executions", color="info", className="ms-2")
        ]),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab([
                    html.Div([
                        dcc.Markdown(response, className="p-3 bg-dark rounded", id="final-response-text")
                    ], className="mt-3")
                ], label="🗨️ Response", tab_id="response-tab"),
                
                dbc.Tab([
                    html.Div([
                        # Search bar for trace filtering
                        dbc.Row([
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.InputGroupText("🔍"),
                                    dbc.Input(
                                        id="trace-search-input",
                                        placeholder="Search agent executions by name or content...",
                                        type="text",
                                        debounce=True
                                    )
                                ])
                            ], width=8),
                            dbc.Col([
                                dbc.Button(
                                    "Clear", 
                                    id="trace-search-clear", 
                                    color="secondary", 
                                    size="sm"
                                )
                            ], width=2),
                            dbc.Col([
                                dbc.Badge(
                                    id="trace-count-badge",
                                    color="info"
                                )
                            ], width=2)
                        ], className="mb-3"),
                        
                        # Grouped tool executions display using same data as analysis
                        html.Div([
                            create_consistent_tool_executions_display(tool_executions, query)
                        ], id="trace-items-container")
                    ], className="mt-3")
                ], label="🤖 Agent Execution", tab_id="trace-tab")
            ], active_tab="response-tab")
        ])
    ])

def extract_grouped_tool_executions(trace_data: List[Dict]) -> List[GroupedToolExecution]:
    """Extract and group tool calls with their returns from proper pydantic_ai messages"""
    tool_executions = []
    call_buffer = {}
    tool_counter = 0
    
    logger.info(f"Extracting grouped tool executions from {len(trace_data)} messages")
    
    for msg_idx, message in enumerate(trace_data):
        if not isinstance(message, dict):
            continue
            
        parts = message.get('parts', [])
        for part_idx, part in enumerate(parts):
            part_kind = part.get('part_kind', 'unknown')
            tool_name = part.get('tool_name', f'Unknown_{msg_idx}_{part_idx}')
            tool_call_id = part.get('tool_call_id', tool_name)
            
            if part_kind == 'tool-call':
                tool_counter += 1
                call_buffer[tool_call_id] = {
                    'tool_number': tool_counter,
                    'tool_name': tool_name,
                    'tool_input': part.get('args', {}),
                    'call_index': len(tool_executions),
                    'has_call': True,
                    'tool_call_id': tool_call_id
                }
                logger.info(f"Buffered tool call #{tool_counter}: {tool_name}")
                
            elif part_kind == 'tool-return':
                if tool_call_id in call_buffer:
                    # Complete execution with both call and return
                    call_data = call_buffer.pop(tool_call_id)
                    execution = GroupedToolExecution(
                        tool_number=call_data['tool_number'],
                        tool_name=call_data['tool_name'],
                        tool_input=call_data['tool_input'],
                        tool_output=part.get('content', {}),
                        has_call=True,
                        has_return=True,
                        tool_call_id=tool_call_id,
                        call_index=call_data['call_index'],
                        return_index=len(tool_executions)
                    )
                    tool_executions.append(execution)
                    logger.info(f"Completed tool execution for: {call_data['tool_name']}")
                else:
                    # Try to match by tool name for orphaned returns
                    matching_call = None
                    for buffered_id, buffered_call in call_buffer.items():
                        if buffered_call['tool_name'] == tool_name:
                            matching_call = buffered_call
                            call_buffer.pop(buffered_id)
                            break
                    
                    if matching_call:
                        execution = GroupedToolExecution(
                            tool_number=matching_call['tool_number'],
                            tool_name=matching_call['tool_name'],
                            tool_input=matching_call['tool_input'],
                            tool_output=part.get('content', {}),
                            has_call=True,
                            has_return=True,
                            tool_call_id=tool_call_id,
                            call_index=matching_call['call_index'],
                            return_index=len(tool_executions)
                        )
                        tool_executions.append(execution)
                        logger.info(f"Matched orphaned return by name: {tool_name}")
                    else:
                        # True orphaned return
                        tool_counter += 1
                        execution = GroupedToolExecution(
                            tool_number=tool_counter,
                            tool_name=tool_name,
                            tool_input="[No matching call found]",
                            tool_output=part.get('content', {}),
                            has_call=False,
                            has_return=True,
                            tool_call_id=tool_call_id,
                            return_index=len(tool_executions)
                        )
                        tool_executions.append(execution)
                        logger.warning(f"Created orphaned return for: {tool_name}")
    
    # Handle remaining orphaned calls
    for tool_call_id, call_data in call_buffer.items():
        execution = GroupedToolExecution(
            tool_number=call_data['tool_number'],
            tool_name=call_data['tool_name'],
            tool_input=call_data['tool_input'],
            tool_output="[No return received]",
            has_call=True,
            has_return=False,
            tool_call_id=tool_call_id,
            call_index=call_data['call_index']
        )
        tool_executions.append(execution)
        logger.warning(f"Created orphaned call for: {call_data['tool_name']}")
    
    # Sort by tool number to maintain execution order
    tool_executions.sort(key=lambda x: x.tool_number)
    
    logger.info(f"Extracted {len(tool_executions)} grouped tool executions")
    return tool_executions

def convert_tool_calls_to_executions(tool_calls: List[ToolCall]) -> List[GroupedToolExecution]:
    """Convert ToolCall objects to GroupedToolExecution objects for consistent display"""
    logger.info(f"[DEBUG] Converting {len(tool_calls)} ToolCall objects to GroupedToolExecution")
    executions = []
    
    for tc in tool_calls:
        # Use tool_idx + 1 to match the numbering in Enhanced Influence Analysis
        tool_number = tc.tool_idx + 1
        logger.info(f"[DEBUG] Converting ToolCall: {tc.tool_name} -> agent: {tc.agent_name} to GroupedToolExecution")
        
        execution = GroupedToolExecution(
            tool_number=tool_number,
            tool_name=tc.tool_name,
            tool_input=tc.tool_input,
            tool_output=tc.tool_output,
            has_call=True,  # Assume all ToolCall objects have both call and return
            has_return=True,
            tool_call_id=tc.tool_call_id,
            call_index=tc.tool_idx,
            return_index=tc.tool_idx,
            agent_name=tc.agent_name,
            agent_model=tc.agent_model
        )
        executions.append(execution)
    
    logger.info(f"Converted {len(tool_calls)} ToolCall objects to GroupedToolExecution objects")
    return executions

def create_consistent_tool_executions_display(tool_executions: List[GroupedToolExecution], query: str) -> html.Div:
    """Create display for tool executions grouped by agent in tree structure"""
    if not tool_executions:
        return html.P("No tool executions found.", className="text-muted")
    
    # Add user query as starting point
    accordion_items = []
    if query:
        query_item = html.Div([
            dbc.AccordionItem(
                html.Div([
                    dbc.Alert("Starting Point: User Query", color="info", className="mb-2"),
                    dbc.Card([
                        dbc.CardBody([
                            html.Pre(query, style={
                                'whiteSpace': 'pre-wrap',
                                'fontSize': '0.9rem'
                            })
                        ])
                    ])
                ]),
                title="🚀 User Query (Starting Point)",
                id="user-query-step"
            )
        ], 
        className="trace-item",
        **{
            'data-search-content': f"user query {query.lower()}",
            'data-tool-number': '0'
        })
        accordion_items.append(query_item)
    
    # Group tool executions by agent
    agent_groups = {}
    for execution in tool_executions:
        agent_name = execution.agent_name or "master_coordinator"
        if agent_name not in agent_groups:
            agent_groups[agent_name] = []
        agent_groups[agent_name].append(execution)
    
    # Create tree structure with agents as top level
    for agent_name, agent_executions in agent_groups.items():
        agent_tools = []
        
        # Process each tool execution for this agent
        for execution in agent_executions:
            tool_number = execution.tool_number
            tool_name = execution.tool_name
            
            # Create status indicators
            status_badges = []
            if execution.has_call:
                status_badges.append(dbc.Badge("CALL", color="primary", className="me-1"))
            if execution.has_return:
                status_badges.append(dbc.Badge("RETURN", color="success", className="me-1"))
            if not execution.has_call:
                status_badges.append(dbc.Badge("ORPHANED RETURN", color="warning", className="me-1"))
            if not execution.has_return:
                status_badges.append(dbc.Badge("NO RETURN", color="secondary", className="me-1"))
            
            # Create tool content
            tool_content = [
                # Status and debug info
                dbc.Alert([
                    html.Strong(f"Tool #{tool_number} Status: "),
                    *status_badges,
                    html.Br(),
                    html.Small(f"Tool Name: {tool_name} | Call ID: {execution.tool_call_id} | Agent: {execution.agent_name or 'master_coordinator'}")
                ], color="light", className="mb-3"),
            ]
            
            # Tool Input Section
            if execution.has_call:
                tool_content.extend([
                    html.H6("🔧 Tool Input:", className="fw-bold mt-2 mb-2"),
                    dbc.Card([
                        dbc.CardHeader(dbc.Badge("INPUT", color="primary")),
                        dbc.CardBody([
                            html.Pre(
                                to_string_for_pre(execution.tool_input),
                                style={
                                    'whiteSpace': 'pre-wrap',
                                    'wordBreak': 'break-word',
                                    'maxHeight': '300px',
                                    'overflowY': 'auto',
                                    'fontSize': '0.85rem'
                                }
                            )
                        ])
                    ], className="mb-3")
                ])
            
            # Tool Output Section
            if execution.has_return:
                tool_content.extend([
                    html.H6("📤 Tool Output:", className="fw-bold mt-2 mb-2"),
                    dbc.Card([
                        dbc.CardHeader(dbc.Badge("OUTPUT", color="success")),
                        dbc.CardBody([
                            html.Pre(
                                to_string_for_pre(execution.tool_output),
                                style={
                                    'whiteSpace': 'pre-wrap',
                                    'wordBreak': 'break-word',
                                    'maxHeight': '400px',
                                    'overflowY': 'auto',
                                    'fontSize': '0.85rem'
                                }
                            )
                        ])
                    ])
                ])
            
            # Add tool to agent's tools list
            agent_tools.append(
                dbc.AccordionItem(
                    html.Div(tool_content),
                    title=f"🔧 Tool #{tool_number}: {tool_name}",
                    id=f"tool-execution-{tool_number}"
                )
            )
        
        # Create agent accordion item with nested tool accordion
        # Get agent emoji dynamically from registry
        try:
            agent_emoji = AGENT_REGISTRY.get(agent_name, {}).get("emoji", "🤖")
        except:
            agent_emoji = "🤖"
        agent_accordion_item = html.Div([
            dbc.AccordionItem(
                html.Div([
                    dbc.Alert([
                        html.Strong(f"Agent: {agent_name}"),
                        html.Br(),
                        html.Small(f"{len(agent_executions)} tool executions")
                    ], color="info", className="mb-3"),
                    dbc.Accordion(
                        agent_tools,
                        id=f"agent-{agent_name}-tools-accordion",
                        start_collapsed=False,
                        always_open=True
                    )
                ]),
                title=f"{agent_emoji} {agent_name} ({len(agent_executions)} tools)",
                id=f"agent-{agent_name}"
            )
        ], className="trace-item")
        
        accordion_items.append(agent_accordion_item)
    
    # Create summary
    total_executions = len(tool_executions)
    complete_executions = len([ex for ex in tool_executions if ex.has_call and ex.has_return])
    orphaned_calls = len([ex for ex in tool_executions if ex.has_call and not ex.has_return])
    orphaned_returns = len([ex for ex in tool_executions if not ex.has_call and ex.has_return])
    
    return html.Div([
        # Summary section
        dbc.Alert([
            html.H6("📊 Agent Execution Summary", className="fw-bold mb-2"),
            html.P([
                f"Total executions: {total_executions} | ",
                f"Complete: {complete_executions} | ",
                f"Agents: {len(agent_groups)}"
            ], className="mb-2"),
            html.P([
                html.Strong("Tree Structure: "),
                "Each agent is shown with its tools grouped underneath. ",
                "Tool numbers match Enhanced Influence Analysis for cross-reference."
            ], className="mb-0 small text-muted")
        ], color="white", className="mb-3"),
        
        html.H6("🌳 Agent Execution Tree", className="fw-bold mb-3"),
        html.P("Agents are shown as top-level items with their tools nested underneath:", 
               className="text-muted mb-3"),
        
        dbc.Accordion(
            accordion_items,
            id="grouped-tools-accordion",
            start_collapsed=False,
            always_open=True
        )
    ])

# Clear search callback
@app.callback(
    Output("trace-search-input", "value"),
    Input("trace-search-clear", "n_clicks"),
    prevent_initial_call=True
)
def clear_trace_search(n_clicks):
    if n_clicks:
        return ""
    raise PreventUpdate

# Fixed clientside callback for search filtering and highlighting
app.clientside_callback(
"""
function(search_value) {
    // Correctly targets the accordion generated by the final python function definition
    const accordion = document.getElementById('grouped-tools-accordion');
    if (!accordion) return '0 items';

    const items = accordion.querySelectorAll('.trace-item');
    const searchTerm = search_value ? search_value.toLowerCase() : '';
    let visibleCount = 0;
    
    // Helper function to escape regex characters for safe use in new RegExp()
    function escapeRegExp(string) {
        if (!string) return '';
        // Escape characters with special meaning in regular expressions
        return string.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
    }
    
    // Helper function to remove existing <mark> highlights from an element
    function removeHighlight(element) {
        if (!element) return;
        const marks = element.querySelectorAll('mark');
        marks.forEach(function(mark) {
            const parent = mark.parentNode;
            if (parent) {
                // Replace the <mark> element with its own text content
                parent.replaceChild(document.createTextNode(mark.textContent), mark);
                // Merge adjacent text nodes for a clean DOM
                parent.normalize();
            }
        });
    }

    // Helper function to add new highlights to text nodes within an element
    function highlightText(element, term) {
        if (!element || !term) return;
        
        // Clear any previous highlights first
        removeHighlight(element);
        
        try {
            const escapedTerm = escapeRegExp(term);
            if (!escapedTerm) return; // Don't highlight if term is empty
            
            const regex = new RegExp(`(${escapedTerm})`, 'gi');
            
            // Use TreeWalker to efficiently find all text nodes
            const treeWalker = document.createTreeWalker(
                element, 
                NodeFilter.SHOW_TEXT, 
                null, 
                false
            );
            
            let textNode;
            const nodesToReplace = [];

            // First, collect all text nodes that contain the search term
            while (textNode = treeWalker.nextNode()) {
                if (textNode.textContent && textNode.textContent.toLowerCase().includes(term)) {
                    nodesToReplace.push(textNode);
                }
            }
            
            // Now, process the collected nodes to avoid issues with a live NodeList
            nodesToReplace.forEach(function(node) {
                try {
                    const fragment = document.createDocumentFragment();
                    const parts = node.textContent.split(regex);
                    
                    for (let i = 0; i < parts.length; i++) {
                        if (i % 2 === 1) { // This is the matched part
                            const mark = document.createElement('mark');
                            mark.style.backgroundColor = '#ffeb3b';
                            mark.style.color = '#000';
                            mark.textContent = parts[i];
                            fragment.appendChild(mark);
                        } else if (parts[i]) { // This is a non-matched part
                            fragment.appendChild(document.createTextNode(parts[i]));
                        }
                    }
                    
                    if (node.parentNode) {
                        node.parentNode.replaceChild(fragment, node);
                    }
                } catch (e) {
                    console.warn('Error during text highlighting replacement:', e);
                }
            });
        } catch (e) {
            console.warn('Error creating RegExp or TreeWalker for highlighting:', e);
        }
    }

    // Main logic to filter and highlight each trace item
    items.forEach(function(item) {
        try {
            const searchContent = item.getAttribute('data-search-content') || '';
            const isMatch = !searchTerm || searchContent.includes(searchTerm);
            
            // Show or hide the item based on the search term
            item.style.display = isMatch ? '' : 'none';
            
            if (isMatch) {
                visibleCount++;
                // Add highlights if there is a search term
                if (searchTerm) {
                    highlightText(item, searchTerm);
                }
            } else {
                // Always remove highlights from hidden items
                removeHighlight(item);
            }
        } catch (e) {
            console.warn('Error processing trace item for search:', e);
            // Keep item visible if there's an error during processing
            item.style.display = '';
            visibleCount++;
        }
    });
    
    // Update the badge with the count of visible items
    return `${visibleCount} visible`;
}
""",
    Output("trace-count-badge", "children"),
    Input("trace-search-input", "value"),
    prevent_initial_call=True
)

def create_enhanced_analysis_layout(analysis_results: Dict) -> html.Div:
    """Create enhanced analysis layout with improved UI and debugging info"""
    selected_text = analysis_results['selected_text']
    best_matches = analysis_results['best_matches']
    influence_chains = analysis_results['influence_chains']
    content_analysis = analysis_results.get('content_analysis', {})
    
    # Build layout children dynamically
    layout_children = [
        # Header with content origin analysis
        html.Div([
            html.H6("🧠 Multi-Agent RoPE Analysis", className="fw-bold mb-0"),
            dbc.Badge("3072D + Agent Tracking + O(n²) Cascaded Analysis", color="success", className="ms-2")
        ], className="d-flex align-items-center mb-3"),
        
        # Content origin analysis
        html.Div([
            html.H6("🔍 Content Origin Analysis", className="fw-bold mb-2"),
            dbc.Badge(
                "LLM Generated" if content_analysis.get('is_llm_generated') else "Tool Derived", 
                color="info" if content_analysis.get('is_llm_generated') else "success",
                className="me-2"
            ),
            dbc.Badge(f"Confidence: {content_analysis.get('tool_derived_confidence', 0):.3f}", color="secondary")
        ], className="mb-3"),
        
        # Selected text
        html.H6("Selected Text:", className="fw-bold"),
        html.Blockquote(f'"{selected_text}"', className="border-start border-4 border-info ps-2 mb-4"),
        
        # Enhanced tool matches with debugging info
        html.H5("🎯 Tool Matches (Sequential Order)", className="mb-3"),
        html.P("Note: Tool numbers match exactly with those in the Tool Executions tab. Parameter-less tools (no input) are scored based only on output similarities.", 
               className="text-muted small mb-3"),
        dbc.Accordion([
            dbc.AccordionItem([
                create_enhanced_tool_match_content(match)
            ], title=f"{match['debug_trace_ref']} | Score: {match['similarity_score']:.4f}")
            for match in best_matches
        ], start_collapsed=False, always_open=True),
        
        html.Hr(),
        
        # Enhanced influence chains
        html.H5("⛓️ Output-Based Influence Chains", className="mb-3"),
        html.P("Semantic relationships traced through tool output similarities:", 
               className="text-muted mb-3"),
    ]
    
    # Add influence chains if they exist
    for chain in influence_chains:
        chain_viz = create_enhanced_influence_chain_visualization(chain)
        if chain_viz:
            layout_children.append(chain_viz)
    
    return html.Div(layout_children)

def create_enhanced_tool_match_content(match: Dict) -> html.Div:
    """Enhanced tool match content with parameter-less tool handling"""
    try:
        debug_trace_ref = match.get('debug_trace_ref', 'Unknown Tool')
        tool_number = match.get('tool_number', match.get('tool_idx', 0) + 1)
        similarity_score = match.get('similarity_score', 0.0)
        has_meaningful_input = match.get('has_meaningful_input', True)
        tool_name = match.get('tool_name', 'Unknown')
        
        # Get agent information
        agent_name = match.get('agent_name', 'master_coordinator')
        agent_model = match.get('agent_model', 'openai:gpt-4o')
        is_delegation = match.get('is_delegation', False)
        delegated_to = match.get('delegated_to', None)
        
        # FORCE parameter-less detection for known tools at display time
        if tool_name in ['get_current_location', 'get_current_date_time'] or str(match.get('tool_input', '')).strip() in ['{}', 'None', '']:
            has_meaningful_input = False
            print(f"FORCED parameter-less for {tool_name} at display time")
        
        # Get all similarity scores with defaults
        rope_input_sim = match.get('rope_input_sim', 0.0)
        rope_output_sim = match.get('rope_output_sim', 0.0)
        base_input_sim = match.get('base_input_sim', 0.0)
        base_output_sim = match.get('base_output_sim', 0.0)
        text_overlap_score = match.get('text_overlap_score', 0.0)
        
        # FORCE input similarities to 0 for parameter-less tools
        if not has_meaningful_input:
            rope_input_sim = 0.0
            base_input_sim = 0.0
            print(f"FORCED input similarities to 0.0 for parameter-less tool: {tool_name}")
        
        tool_input = match.get('tool_input', 'No input data')
        tool_output = match.get('tool_output', 'No output data')
        
        return html.Div([
            # Debugging and reference info with agent information
            dbc.Alert([
                html.Strong("Cross-Reference: "),
                f"Tool #{tool_number} in Agent Execution tab | ",
                html.Strong("Agent: "),
                f"{agent_name} ({agent_model}) | ",
                html.Strong("Tool: "),
                f"{tool_name} | ",
                html.Strong("Type: "),
                f"{'🔄 Delegation to ' + delegated_to if is_delegation else '🛠️ Direct tool execution'} | ",
                html.Strong("Input: "), 
                f"{'Parameter-less (output-only analysis)' if not has_meaningful_input else 'Has input parameters'}"
            ], color="light" if has_meaningful_input else "warning", className="mb-3"),
            
            # Enhanced similarity breakdown
            dbc.Row([
                dbc.Col([
                    html.H6("Similarity Breakdown:", className="fw-bold mb-2"),
                    dbc.Badge(f"Final: {similarity_score:.4f}", color="primary", className="me-2 mb-1"),
                    # Show output similarities (always available)
                    dbc.Badge(f"RoPE Out: {rope_output_sim:.4f}", color="info", className="me-2 mb-1"),
                    dbc.Badge(f"Base Out: {base_output_sim:.4f}", color="secondary", className="me-2 mb-1"),
                    dbc.Badge(f"Text Overlap: {text_overlap_score:.4f}", color="success", className="me-2 mb-1"),
                    # NEVER show input similarities for parameter-less tools
                    html.Br() if has_meaningful_input and (rope_input_sim > 0.001 and base_input_sim > 0.001) else "",
                    dbc.Badge(f"RoPE In: {rope_input_sim:.4f}", color="info", className="me-2 mb-1") if has_meaningful_input and rope_input_sim > 0.001 else "",
                    dbc.Badge(f"Base In: {base_input_sim:.4f}", color="secondary", className="me-2 mb-1") if has_meaningful_input and base_input_sim > 0.001 else "",
                    # Add explanation for parameter-less tools
                    html.Br() if not has_meaningful_input else "",
                    dbc.Badge("Note: Parameter-less tool - scoring based only on output", color="warning", className="me-2 mb-1") if not has_meaningful_input else "",
                ], width=12)
            ], className="mb-3"),
            
            # Full tool input (no truncation)
            html.Strong("Tool Input (Full):", className="d-block mt-3 mb-2"),
            dbc.Card([
                dbc.CardBody([
                    html.Pre(
                        "[No input parameters - this tool requires no input]" if not has_meaningful_input 
                        else str(tool_input),
                        style={
                            'whiteSpace': 'pre-wrap', 
                            'wordBreak': 'break-word',
                            'fontSize': '0.85rem',
                            'maxHeight': '200px',
                            'overflowY': 'auto',
                            'fontStyle': 'italic' if not has_meaningful_input else 'normal',
                            'color': '#888' if not has_meaningful_input else 'inherit'
                        }
                    )
                ])
            ], className="mb-3"),
            
            # Full tool output (no truncation) 
            html.Strong("Tool Output (Full):", className="d-block mb-2"),
            dbc.Card([
                dbc.CardBody([
                    dcc.Markdown(str(tool_output), style={
                        'maxHeight': '300px',
                        'overflowY': 'auto',
                        'fontSize': '0.85rem'
                    })
                ])
            ])
        ])
        
    except Exception as e:
        logger.error(f"Error creating tool match content: {e}")
        return html.Div([
            dbc.Alert(f"Error displaying tool match: {str(e)}", color="warning")
        ])

def create_enhanced_influence_chain_visualization(chain_data: Dict) -> dbc.Card:
    """Enhanced influence chain visualization"""
    try:
        root_tool = chain_data.get('root_tool', {})
        chain = chain_data.get('chain', {})
        chain_length = chain_data.get('chain_length', 0)
        chain_type = chain_data.get('chain_type', 'output_based')
        
        if not root_tool or not chain:
            return html.Div()  # Return empty div if invalid data
        
        debug_trace_ref = root_tool.get('debug_trace_ref', 'Unknown Tool')
        
        card_content = create_enhanced_chain_level_content(chain, level=0)
        
        return dbc.Card([
            dbc.CardHeader([
                html.H6(f"🔗 {chain_type.title()} Chain", className="mb-0"),
                dbc.Badge(f"{debug_trace_ref}", color="info", className="me-2"),
                dbc.Badge(f"{chain_length} levels", color="secondary")
            ], className="d-flex align-items-center"),
            dbc.CardBody([card_content])
        ], className="mb-3")
        
    except Exception as e:
        logger.error(f"Error creating influence chain visualization: {e}")
        return html.Div()  # Return empty div on error

def create_enhanced_chain_level_content(chain_node: Dict, level: int, max_display_level: int = 4) -> html.Div:
    """Enhanced chain level content with better visualization"""
    if not chain_node or level > max_display_level:
        return html.Div()
    
    colors = ["primary", "secondary", "success", "info", "warning"]
    color = colors[min(level, len(colors) - 1)]
    margin_left = f"{level * 25}px"
    
    # Use proper sequential tool numbering (tool_idx is 0-based, so add 1)
    tool_number = chain_node['tool_idx'] + 1
    
    # Build children list dynamically to avoid None values
    div_children = [
        html.Div([
            dbc.Badge(f"L{level}", color=color, className="me-2"),
            html.Strong(f"Tool #{tool_number}: {chain_node['tool_name']}", className="me-2"),
            dbc.Badge(f"Score: {chain_node['influence_score']:.4f}", color="light", className="text-dark"),
        ], className="d-flex align-items-center mb-2"),
        
        html.P(chain_node['explanation'], className="text-muted small mb-2"),
        
        html.P([
            html.I(className="bi bi-arrow-right me-2"),
            f"Reference: Find Tool #{tool_number} in Tool Executions tab"
        ], className="text-info small mb-2"),
    ]
    
    # Add parent chain indicator if exists
    if chain_node.get('parent_chain'):
        div_children.append(html.Div([
            html.I(className="bi bi-arrow-up-right me-2"),
            "Output influenced by..."
        ], className="text-warning small mb-2"))
    
    # Main div content
    main_div_children = [
        html.Div(div_children, style={
            "marginLeft": margin_left, 
            "paddingLeft": "15px", 
            "borderLeft": f"3px solid var(--bs-{color})",
            "paddingTop": "10px",
            "paddingBottom": "10px"
        })
    ]
    
    # Add parent chain recursively if exists
    if chain_node.get('parent_chain'):
        parent_content = create_enhanced_chain_level_content(
            chain_node.get('parent_chain'), level + 1, max_display_level
        )
        if parent_content:
            main_div_children.append(parent_content)
    
    return html.Div(main_div_children, className="mb-2")

# --- Utility Functions ---

def to_string_for_pre(content: Any) -> str:
    """Convert content to string for display"""
    if content is None:
        return "None"
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return content
    try: 
        return json.dumps(content, indent=2, ensure_ascii=False)
    except (TypeError, ValueError): 
        return str(content)

def ensure_json_serializable(obj: Any) -> Any:
    """Ensure object is JSON serializable"""
    if isinstance(obj, (bytes, bytearray)): 
        return obj.decode('utf-8', 'replace')
    if isinstance(obj, dict): 
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): 
        return [ensure_json_serializable(i) for i in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Travel Agent with Parameter-less Tool Handling')
    parser.add_argument('--env', default=".env", help='Path to .env file')
    parser.add_argument('--port', type=int, default=8050, help='Port number')
    args = parser.parse_args()
    
    if os.path.exists(args.env):
        load_dotenv(args.env)
        logger.info(f"Loaded environment variables from {args.env}")
    else:
        logger.warning(f".env file not found at {args.env}. Using system environment variables.")
    
    app.run(debug=True, host="0.0.0.0", port=args.port)