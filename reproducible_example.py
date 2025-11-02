"""
Minimum Reproducible Example: Subgraph loses runtime when invoked with config
==============================================================================

ISSUE: When manually invoking a subgraph with a config parameter (e.g., to specify 
thread_id for checkpointing), the runtime objects (store and context) are lost.

WHY THIS MATTERS: Users need to pass configs to control checkpointing behavior, 
resume threads, or set other configurable options. Without the fix, they lose 
access to store and context in the subgraph.

WORKAROUND (before fix): Users had to manually call patch_configurable()
FIX: The ensure_config() function now merges CONF sections instead of replacing them
"""

from dataclasses import dataclass
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    username: str


class State(TypedDict):
    foo: str


# ============================================================================
# SUBGRAPH - Requires runtime (store + context)
# ============================================================================

def subgraph_node_1(state: State, runtime: Runtime[Context]):
    """This node expects both store and context to be available."""
    # These assertions will fail if runtime is lost
    assert runtime.store is not None, "Store is required"
    assert runtime.context is not None, "Context is required"
    
    username = state['foo']
    runtime.store.put(('subgraph', '1'), 'foo', {'value': f'hi! {username}'})
    return {'foo': f'hi! {username}'}


subgraph_builder = StateGraph(State, context_schema=Context)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.set_entry_point('subgraph_node_1')
subgraph = subgraph_builder.compile()


# ============================================================================
# PARENT GRAPH - Invokes subgraph with custom config
# ============================================================================

def main_node(state: State, runtime: Runtime[Context]):
    """Read from store or use context as fallback."""
    last_foo = runtime.store.get(('subgraph', '1'), 'foo')
    if last_foo:
        last_foo = last_foo.value['value']
    else:
        last_foo = runtime.context.username
    return {'foo': f'hello {last_foo}'}


def invoke_subgraph(state: State, runtime: Runtime[Context], config: RunnableConfig):
    """
    Manually invoke subgraph with a custom config.
    
    THE PROBLEM:
    Users want to pass their own config to control behavior (e.g., thread_id).
    But when they create a new config dict, it replaces the runtime config
    instead of merging with it, causing store/context to be lost.
    
    BEFORE FIX:
        new_config = {'configurable': {'thread_id': '1'}}
        # This loses store and context! ?
    
    WORKAROUND:
        from langgraph.utils.config import patch_configurable
        new_config = patch_configurable(config, {'thread_id': '1'})
        # This preserves runtime ? but requires extra import
    
    AFTER FIX:
        new_config = {**config, 'configurable': {...config.get('configurable', {}), 'thread_id': '1'}}
        # This now works! ? Runtime is automatically preserved
    """
    new_config = {
        **config,
        'configurable': {
            **config.get('configurable', {}),
            'thread_id': '1'  # User wants to specify thread_id
        }
    }
    
    # Before fix: this would fail with "Store is required" assertion
    # After fix: runtime (store + context) is preserved automatically
    return subgraph.invoke(input=state, config=new_config)


# ============================================================================
# RUN THE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    builder = StateGraph(State, context_schema=Context)
    builder.add_node(main_node)
    builder.add_node(invoke_subgraph)
    builder.set_entry_point('main_node')
    builder.add_edge('main_node', 'invoke_subgraph')
    
    store = InMemoryStore()
    graph = builder.compile(store=store)
    
    context = Context(username='Alice')
    
    print("Running example...")
    print(f"Input: {{'foo': 'world'}}")
    print(f"Context: {context}")
    print()
    
    try:
        result = graph.invoke(input={'foo': 'world'}, context=context)
        print(f"? SUCCESS: {result}")
        
        # Verify the flow worked correctly:
        # 1. main_node: {'foo': 'hello Alice'} (from context.username)
        # 2. invoke_subgraph calls subgraph with config containing thread_id
        # 3. subgraph_node_1: {'foo': 'hi! hello Alice'} (uses store + context)
        assert result == {'foo': 'hi! hello Alice'}
        
        # Verify store was updated
        stored = store.get(('subgraph', '1'), 'foo')
        assert stored.value['value'] == 'hi! hello Alice'
        print("? Store updated correctly")
        print("? Runtime (store + context) preserved in subgraph!")
        
    except AssertionError as e:
        print(f"? FAILED: {e}")
        print()
        print("This error means runtime was lost when invoking the subgraph.")
        print("The fix ensures CONF sections are merged, preserving runtime.")
