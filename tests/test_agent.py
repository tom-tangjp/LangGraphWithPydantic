"""Tests for agent module."""

import pytest
from agent import (
    AgentState,
    AgentStepModel,
    PlanModel,
    IntentModel,
    ReflectModel,
    hash_step,
    current_step,
)


class TestAgentStepModel:
    """Tests for AgentStepModel."""

    def test_agent_step_creation(self):
        """Test creating an agent step."""
        step = AgentStepModel(
            id="s1",
            title="Test Step",
            agent="researcher",
            task="Perform a test task",
            acceptance="Task completed",
        )
        assert step.id == "s1"
        assert step.agent == "researcher"
        assert step.task == "Perform a test task"

    def test_agent_step_with_dependencies(self):
        """Test agent step with dependencies."""
        step = AgentStepModel(
            id="s2",
            title="Dependent Step",
            agent="solver",
            task="Solve the problem",
            acceptance="Solution provided",
            depends_on=["s1"],
        )
        assert "s1" in step.depends_on


class TestPlanModel:
    """Tests for PlanModel."""

    def test_plan_creation(self):
        """Test creating a plan."""
        plan = PlanModel(
            version=1,
            objective="Test objective",
            steps=[
                AgentStepModel(
                    id="s1",
                    title="Step 1",
                    agent="researcher",
                    task="Research",
                    acceptance="Research done",
                )
            ],
        )
        assert plan.version == 1
        assert len(plan.steps) == 1

    def test_plan_with_multiple_steps(self):
        """Test plan with multiple steps."""
        steps = [
            AgentStepModel(
                id=f"s{i}",
                title=f"Step {i}",
                agent="researcher" if i == 0 else "solver",
                task=f"Task {i}",
                acceptance=f"Done {i}",
            )
            for i in range(3)
        ]
        plan = PlanModel(version=1, objective="Multi-step plan", steps=steps)
        assert len(plan.steps) == 3


class TestIntentModel:
    """Tests for IntentModel."""

    def test_intent_creation(self):
        """Test creating an intent."""
        intent = IntentModel(
            task_type="research",
            user_goal="Research AI safety",
            domains=["technology"],
            deliverable="report",
            need_web=True,
            missing_info=[],
        )
        assert intent.task_type == "research"
        assert intent.user_goal == "Research AI safety"

    def test_intent_with_entities(self):
        """Test intent with entities."""
        from agent import IntentEntity

        intent = IntentModel(
            task_type="analysis",
            user_goal="Analyze performance",
            domains=["software"],
            deliverable="report",
            missing_info=[],
            entities=[
                IntentEntity(type="codebase", value="main.py"),
                IntentEntity(type="file", value="/path/to/data.csv"),
            ],
        )
        assert len(intent.entities) == 2


class TestReflectModel:
    """Tests for ReflectModel."""

    def test_reflect_accept(self):
        """Test reflect with accept decision."""
        result = ReflectModel(decision="accept", reason="Good work")
        assert result.decision == "accept"

    def test_reflect_retry(self):
        """Test reflect with retry decision."""
        result = ReflectModel(
            decision="retry",
            reason="Incomplete",
            required_changes=["Add more details"],
        )
        assert result.decision == "retry"
        assert "Add more details" in result.required_changes


class TestHashStep:
    """Tests for hash_step function."""

    def test_hash_step_consistency(self):
        """Test that same step produces same hash."""
        step = {
            "id": "s1",
            "title": "Test",
            "agent": "researcher",
            "task": "Task",
            "acceptance": "Done",
            "inputs": {},
            "depends_on": [],
        }
        hash1 = hash_step(step)
        hash2 = hash_step(step)
        assert hash1 == hash2

    def test_hash_step_different_inputs(self):
        """Test that different steps produce different hashes."""
        step1 = {
            "id": "s1",
            "title": "Test 1",
            "agent": "researcher",
            "task": "Task 1",
            "acceptance": "Done",
            "inputs": {},
            "depends_on": [],
        }
        step2 = {
            "id": "s2",
            "title": "Test 2",
            "agent": "solver",
            "task": "Task 2",
            "acceptance": "Done",
            "inputs": {},
            "depends_on": [],
        }
        hash1 = hash_step(step1)
        hash2 = hash_step(step2)
        assert hash1 != hash2


class TestCurrentStep:
    """Tests for current_step function."""

    def test_current_step(self):
        """Test getting current step from state."""
        plan = {
            "steps": [
                {"id": "s1", "title": "Step 1"},
                {"id": "s2", "title": "Step 2"},
            ]
        }
        state = {"plan": plan, "step_idx": 1}
        step = current_step(state)
        assert step["id"] == "s2"

    def test_current_step_first(self):
        """Test getting first step."""
        plan = {"steps": [{"id": "s1", "title": "First"}]}
        state = {"plan": plan, "step_idx": 0}
        step = current_step(state)
        assert step["id"] == "s1"


class TestAgentState:
    """Tests for AgentState TypedDict."""

    def test_agent_state_creation(self):
        """Test creating an agent state."""
        state: AgentState = {
            "user_request": "Test request",
            "messages": [],
            "artifacts": {},
            "reflections": [],
            "seen_step_hashes": [],
            "step_idx": 0,
            "iter_count": 0,
            "max_iters": 20,
            "final_answer": "",
            "done": False,
            "step_failures": {},
            "last_feedback": {},
            "last_output_hash": {},
            "no_progress": {},
        }
        assert state["user_request"] == "Test request"
        assert state["step_idx"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
