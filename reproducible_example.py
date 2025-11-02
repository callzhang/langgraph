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

# Subgraph
def subgraph_node_1(state: State, runtime: Runtime[Context]):
    assert runtime.store is not None, "Store is required"
    assert runtime.context is not None, "Context is required"
    username = state['foo']
    runtime.store.put(('subgraph', '1'), 'foo', {'value': 'hi! ' + username})
    return {'foo': 'hi! ' + username}

subgraph_builder = StateGraph(State, context_schema=Context)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.set_entry_point('subgraph_node_1')
subgraph = subgraph_builder.compile()

# Parent graph
def main_node(state: State, runtime: Runtime[Context]):
    last_foo = runtime.store.get(('subgraph', '1'), 'foo')
    if last_foo:
        last_foo = last_foo.value['value']
    else:
        last_foo = runtime.context.username
    return {'foo': 'hello ' + str(last_foo)}

def invoke_subgraph(state: State, runtime: Runtime[Context], config: RunnableConfig):
    new_config = {
        **config,
        'configurable': {
            **config.get('configurable', {}),
            'thread_id': '1'
        },
    }
    return subgraph.invoke(input=state, config=new_config)

store = InMemoryStore()
builder = StateGraph(State, context_schema=Context)
builder.add_node(main_node)
builder.add_node(invoke_subgraph)
builder.set_entry_point('main_node')
builder.add_edge('main_node', 'invoke_subgraph')
graph = builder.compile(store=store)

if __name__ == "__main__":
    context = Context(username='Alice')
    result = graph.invoke(input={'foo': 'world'}, context=context)
    print(result)
