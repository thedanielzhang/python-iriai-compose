import iriai_compose


def test_all_public_names_importable():
    expected = [
        "Actor", "AgentActor", "InteractionActor", "Role",
        "Task", "Ask", "Interview", "Gate", "Choose", "Respond",
        "Phase", "Workflow", "Feature", "Workspace",
        "WorkflowRunner", "DefaultWorkflowRunner", "AgentRuntime", "InteractionRuntime",
        "Pending",
        "ArtifactStore", "SessionStore", "AgentSession", "ContextProvider",
        "InMemoryArtifactStore", "InMemorySessionStore", "DefaultContextProvider",
        "IriaiError", "ResolutionError", "TaskExecutionError",
    ]
    for name in expected:
        assert hasattr(iriai_compose, name), f"{name} not found in iriai_compose"


def test_all_matches_expected():
    expected = {
        "Actor", "AgentActor", "InteractionActor", "Role",
        "Task", "Ask", "Interview", "Gate", "Choose", "Respond",
        "Phase", "Workflow", "Feature", "Workspace",
        "WorkflowRunner", "DefaultWorkflowRunner", "AgentRuntime", "InteractionRuntime",
        "Pending",
        "ArtifactStore", "SessionStore", "AgentSession", "ContextProvider",
        "InMemoryArtifactStore", "InMemorySessionStore", "DefaultContextProvider",
        "IriaiError", "ResolutionError", "TaskExecutionError",
        "to_str",
    }
    assert set(iriai_compose.__all__) == expected


def test_runtimes_importable():
    from iriai_compose.runtimes import AutoApproveRuntime, TerminalInteractionRuntime
    assert AutoApproveRuntime is not None
    assert TerminalInteractionRuntime is not None


def test_claude_runtime_moved_to_build_v2():
    """ClaudeAgentRuntime has been moved to iriai-build-v2."""
    import importlib
    assert importlib.util.find_spec("iriai_compose.runtimes.claude") is None
