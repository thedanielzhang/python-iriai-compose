import warnings

import pytest
from pathlib import Path

from iriai_compose import (
    AgentSession,
    ArtifactStore,
    DefaultContextProvider,
    Feature,
    InMemoryArtifactStore,
    InMemorySessionStore,
    InMemoryStore,
    Store,
)


@pytest.fixture
def feature():
    return Feature(
        id="f1", name="F1", slug="f1", workflow_name="test", workspace_id="main"
    )


@pytest.fixture
def feature2():
    return Feature(
        id="f2", name="F2", slug="f2", workflow_name="test", workspace_id="main"
    )


# --- Store / InMemoryStore ---


def test_artifact_store_is_subclass_of_store():
    assert issubclass(ArtifactStore, Store)


def test_in_memory_artifact_store_is_in_memory_store():
    assert InMemoryArtifactStore is InMemoryStore


async def test_store_put_get(feature):
    store = InMemoryStore()
    await store.put("prd", {"content": "PRD data"}, feature=feature)
    result = await store.get("prd", feature=feature)
    assert result == {"content": "PRD data"}


async def test_store_missing_key(feature):
    store = InMemoryStore()
    result = await store.get("nonexistent", feature=feature)
    assert result is None


async def test_store_feature_isolation(feature, feature2):
    store = InMemoryStore()
    await store.put("prd", "f1-prd", feature=feature)
    await store.put("prd", "f2-prd", feature=feature2)
    assert await store.get("prd", feature=feature) == "f1-prd"
    assert await store.get("prd", feature=feature2) == "f2-prd"


async def test_store_overwrite(feature):
    store = InMemoryStore()
    await store.put("prd", "v1", feature=feature)
    await store.put("prd", "v2", feature=feature)
    assert await store.get("prd", feature=feature) == "v2"


async def test_store_delete(feature):
    store = InMemoryStore()
    await store.put("prd", "data", feature=feature)
    await store.delete("prd", feature=feature)
    assert await store.get("prd", feature=feature) is None


# --- Backward compat: InMemoryArtifactStore ---


async def test_artifact_put_get(feature):
    store = InMemoryArtifactStore()
    await store.put("prd", {"content": "PRD data"}, feature=feature)
    result = await store.get("prd", feature=feature)
    assert result == {"content": "PRD data"}


async def test_artifact_missing_key(feature):
    store = InMemoryArtifactStore()
    result = await store.get("nonexistent", feature=feature)
    assert result is None


async def test_artifact_feature_isolation(feature, feature2):
    store = InMemoryArtifactStore()
    await store.put("prd", "f1-prd", feature=feature)
    await store.put("prd", "f2-prd", feature=feature2)
    assert await store.get("prd", feature=feature) == "f1-prd"
    assert await store.get("prd", feature=feature2) == "f2-prd"


async def test_artifact_overwrite(feature):
    store = InMemoryArtifactStore()
    await store.put("prd", "v1", feature=feature)
    await store.put("prd", "v2", feature=feature)
    assert await store.get("prd", feature=feature) == "v2"


# --- SessionStore (deprecated but still functional) ---


async def test_session_store():
    store = InMemorySessionStore()
    session = AgentSession(session_key="pm:f1", session_id="s123")
    await store.save(session)
    loaded = await store.load("pm:f1")
    assert loaded is not None
    assert loaded.session_id == "s123"


async def test_session_store_missing():
    store = InMemorySessionStore()
    assert await store.load("nonexistent") is None


# --- DefaultContextProvider with stores= ---


async def test_context_provider_stores_kwarg(feature):
    store = InMemoryStore()
    await store.put("prd", "The PRD content", feature=feature)
    await store.put("design", "The design content", feature=feature)
    provider = DefaultContextProvider(stores={"artifacts": store})
    result = await provider.resolve(["prd", "design"], feature=feature)
    assert "## prd" in result
    assert "The PRD content" in result
    assert "## design" in result
    assert "The design content" in result


async def test_context_provider_multiple_stores(feature):
    store_a = InMemoryStore()
    store_b = InMemoryStore()
    await store_a.put("prd", "PRD from A", feature=feature)
    await store_b.put("events", "Events from B", feature=feature)
    provider = DefaultContextProvider(stores={"a": store_a, "b": store_b})
    result = await provider.resolve(["prd", "events"], feature=feature)
    assert "PRD from A" in result
    assert "Events from B" in result


async def test_context_provider_multiple_stores_first_wins(feature):
    """When the same key exists in multiple stores, first store wins."""
    store_a = InMemoryStore()
    store_b = InMemoryStore()
    await store_a.put("data", "from A", feature=feature)
    await store_b.put("data", "from B", feature=feature)
    provider = DefaultContextProvider(stores={"a": store_a, "b": store_b})
    result = await provider.resolve(["data"], feature=feature)
    assert "from A" in result
    assert "from B" not in result


async def test_context_provider_missing_keys_skipped(feature):
    store = InMemoryStore()
    await store.put("prd", "PRD", feature=feature)
    provider = DefaultContextProvider(stores={"artifacts": store})
    result = await provider.resolve(["prd", "nonexistent"], feature=feature)
    assert "## prd" in result
    assert "nonexistent" not in result


async def test_context_provider_static_files(feature, tmp_path):
    store = InMemoryStore()
    static_file = tmp_path / "project.md"
    static_file.write_text("Project description")
    provider = DefaultContextProvider(
        stores={"artifacts": store}, static_files={"project": static_file}
    )
    result = await provider.resolve(["project"], feature=feature)
    assert "## project" in result
    assert "Project description" in result


async def test_context_provider_empty_keys(feature):
    store = InMemoryStore()
    provider = DefaultContextProvider(stores={"artifacts": store})
    result = await provider.resolve([], feature=feature)
    assert result == ""


async def test_context_provider_no_stores(feature):
    provider = DefaultContextProvider()
    result = await provider.resolve(["anything"], feature=feature)
    assert result == ""


# --- DefaultContextProvider backward compat: artifacts= ---


async def test_context_provider_artifacts_kwarg_deprecated(feature):
    store = InMemoryStore()
    await store.put("prd", "The PRD content", feature=feature)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        provider = DefaultContextProvider(artifacts=store)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "artifacts=" in str(w[0].message)
    result = await provider.resolve(["prd"], feature=feature)
    assert "The PRD content" in result
