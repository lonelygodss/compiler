#=======================visualization=========================
import graphviz
from typing import Dict, Set, List
from model_compiler.utils import OperationType,TensorId,TensorWithSize,Function,SubFunction,Model,CompiledModel



def visualize_compiled_model(compiled_model: CompiledModel, filename: str = "compiled_model"):
    """
    Visualize the compiled model as a compute graph using Graphviz with shorter tensor labels
    
    Args:
        compiled_model: The compiled model to visualize
        filename: Base filename for the output (without extension)
    """
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Compiled Model', format='pdf')
    dot.attr(rankdir='TB', size='11,8', ratio='fill', concentrate='true')
    
    # Track tensors that have been added to the graph
    added_tensors = set()
    
    # Track subfunctions by operation type for layout
    mvm_subfuncs = {}  # Dict[(k,m,n), Dict[(i,j), SubFunction]]
    other_subfuncs = {}  # Dict[(op_type, k, m, n), List[SubFunction]]
    
    # Group subfunctions by type and coordinates
    for subfunc in compiled_model.subfunctions:
        if subfunc.op_type == OperationType.MVM:
            key = (subfunc.k, subfunc.m, subfunc.n)
            if key not in mvm_subfuncs:
                mvm_subfuncs[key] = {}
            mvm_subfuncs[key][(subfunc.i, subfunc.j)] = subfunc
        else:
            key = (subfunc.op_type, subfunc.k, subfunc.m, subfunc.n)
            if key not in other_subfuncs:
                other_subfuncs[key] = []
            other_subfuncs[key].append(subfunc)
    
    # Define colors for different operation types
    op_colors = {
        OperationType.MVM: "lightblue",
        OperationType.ACTIVATION: "lightgreen",
        OperationType.TRIVIAL_COPY: "lightyellow",
        OperationType.DOT_PRODUCT: "lightpink",
        OperationType.GLU: "orange",
        OperationType.ADD: "salmon",
        OperationType.CONCAT: "violet",
        OperationType.DISTRIBUTE: "cyan",
        OperationType.PASS: "gray"
    }
    
    # Helper function to create a short tensor label
    def get_short_tensor_label(tensor_id):
        if tensor_id.i is not None and tensor_id.j is not None:
            return f"T{tensor_id.k}.{tensor_id.m}.{tensor_id.n}({tensor_id.i},{tensor_id.j})"
        else:
            return f"T{tensor_id.k}.{tensor_id.m}.{tensor_id.n}"
    
    # Helper function to add a tensor node if it hasn't been added yet
    def add_tensor_node(tensor_id, size_info=""):
        tensor_str = str(tensor_id)  # Use full string as node ID
        short_label = get_short_tensor_label(tensor_id)
        
        if tensor_id not in added_tensors:
            label = f"{short_label}{size_info}"
            dot.node(tensor_str, label, shape="ellipse", style="filled", fillcolor="lightgray")
            added_tensors.add(tensor_id)
    
    # Helper function to add a subfunction node
    def add_subfunction_node(subfunc, cluster=None):
        subfunc_id = f"func_{subfunc.i}_{subfunc.j}_{subfunc.k}_{subfunc.m}_{subfunc.n}_{subfunc.op_type.value}"
        
        # Create shorter label for subfunctions
        op_abbr = ''.join([c for c in subfunc.op_type.value if c.isupper()])
        if not op_abbr:
            op_abbr = subfunc.op_type.value[:3]
            
        label = f"{op_abbr}\n({subfunc.i},{subfunc.j},{subfunc.k},{subfunc.m},{subfunc.n})"
        
        if subfunc.shape:
            label += f"\n{subfunc.shape[0]}×{subfunc.shape[1]}"
        
        if cluster:
            cluster.node(subfunc_id, label, shape="box", style="filled", 
                         fillcolor=op_colors.get(subfunc.op_type, "white"))
        else:
            dot.node(subfunc_id, label, shape="box", style="filled", 
                     fillcolor=op_colors.get(subfunc.op_type, "white"))
        
        # Add edges from input tensors to this subfunction
        for input_tensor in subfunc.input_tensors:
            tensor_id = input_tensor.tensor_id
            tensor_str = str(tensor_id)
            size_info = ""
            if input_tensor.size_h is not None and input_tensor.size_v is not None:
                size_info = f"\n{input_tensor.size_h}×{input_tensor.size_v}"
            add_tensor_node(tensor_id, size_info)
            dot.edge(tensor_str, subfunc_id)
        
        # Add edges from this subfunction to its output tensors
        for output_tensor in subfunc.output_tensor:
            tensor_id = output_tensor.tensor_id
            tensor_str = str(tensor_id)
            size_info = ""
            if output_tensor.size_h is not None and output_tensor.size_v is not None:
                size_info = f"\n{output_tensor.size_h}×{output_tensor.size_v}"
            add_tensor_node(tensor_id, size_info)
            dot.edge(subfunc_id, tensor_str)
        
        return subfunc_id
    
    # Process MVM operations in matrix layout
    for (k, m, n), subfuncs_by_ij in mvm_subfuncs.items():
        # Create a subgraph cluster for this MVM operation
        with dot.subgraph(name=f"cluster_mvm_{k}_{m}_{n}") as c:
            c.attr(label=f"MVM (k={k}, m={m}, n={n})", style="filled", color="lightgrey")
            
            # Add MVM subfunctions in a grid
            for (i, j), subfunc in subfuncs_by_ij.items():
                add_subfunction_node(subfunc, c)
    
    # Process other operations
    for (op_type, k, m, n), subfuncs in other_subfuncs.items():
        # Create a subgraph cluster for this operation type
        with dot.subgraph(name=f"cluster_{op_type.value}_{k}_{m}_{n}") as c:
            c.attr(label=f"{op_type.value} (k={k}, m={m}, n={n})", style="filled", color="lightgrey")
            
            # Add subfunctions
            for subfunc in subfuncs:
                add_subfunction_node(subfunc, c)
    
    # Render the graph
    dot.render(filename, view=True)
    print(f"Graph saved as {filename}.pdf")

def visualize_compiled_model_layered(compiled_model: CompiledModel, filename: str = "compiled_model_layered"):
    """
    Visualize the compiled model as a layered compute graph using Graphviz with shorter tensor labels
    
    Args:
        compiled_model: The compiled model to visualize
        filename: Base filename for the output (without extension)
    """
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Compiled Model (Layered)', format='pdf')
    dot.attr(rankdir='TB', size='11,8', ratio='fill')
    
    # Track tensors that have been added to the graph
    added_tensors = set()
    
    # Group subfunctions by m (sequence order) for layered layout
    subfuncs_by_m = {}
    for subfunc in compiled_model.subfunctions:
        if subfunc.m not in subfuncs_by_m:
            subfuncs_by_m[subfunc.m] = []
        subfuncs_by_m[subfunc.m].append(subfunc)
    
    # Define colors for different operation types
    op_colors = {
        OperationType.MVM: "lightblue",
        OperationType.ACTIVATION: "lightgreen",
        OperationType.TRIVIAL_COPY: "lightyellow",
        OperationType.DOT_PRODUCT: "lightpink",
        OperationType.GLU: "orange",
        OperationType.ADD: "salmon",
        OperationType.CONCAT: "violet",
        OperationType.DISTRIBUTE: "cyan",
        OperationType.PASS: "gray"
    }
    
    # Helper function to create a short tensor label
    def get_short_tensor_label(tensor_id):
        if tensor_id.i is not None and tensor_id.j is not None:
            return f"T{tensor_id.k}.{tensor_id.m}.{tensor_id.n}({tensor_id.i},{tensor_id.j})"
        else:
            return f"T{tensor_id.k}.{tensor_id.m}.{tensor_id.n}"
    
    # Helper function to add a tensor node if it hasn't been added yet
    def add_tensor_node(tensor_id, size_info=""):
        tensor_str = str(tensor_id)  # Use full string as node ID
        short_label = get_short_tensor_label(tensor_id)
        
        if tensor_id not in added_tensors:
            label = f"{short_label}{size_info}"
            dot.node(tensor_str, label, shape="ellipse", style="filled", fillcolor="lightgray")
            added_tensors.add(tensor_id)
    
    # Helper function to add a subfunction node
    def add_subfunction_node(subfunc, cluster=None):
        subfunc_id = f"func_{subfunc.i}_{subfunc.j}_{subfunc.k}_{subfunc.m}_{subfunc.n}_{subfunc.op_type.value}"
        
        # Create shorter label for subfunctions
        op_abbr = ''.join([c for c in subfunc.op_type.value if c.isupper()])
        if not op_abbr:
            op_abbr = subfunc.op_type.value[:3]
            
        label = f"{op_abbr}\n({subfunc.i},{subfunc.j},{subfunc.k},{subfunc.m},{subfunc.n})"
        
        if subfunc.shape:
            label += f"\n{subfunc.shape[0]}×{subfunc.shape[1]}"
        
        if cluster:
            cluster.node(subfunc_id, label, shape="box", style="filled", 
                         fillcolor=op_colors.get(subfunc.op_type, "white"))
        else:
            dot.node(subfunc_id, label, shape="box", style="filled", 
                     fillcolor=op_colors.get(subfunc.op_type, "white"))
        
        # Add edges from input tensors to this subfunction
        for input_tensor in subfunc.input_tensors:
            tensor_id = input_tensor.tensor_id
            tensor_str = str(tensor_id)
            size_info = ""
            if input_tensor.size_h is not None and input_tensor.size_v is not None:
                size_info = f"\n{input_tensor.size_h}×{input_tensor.size_v}"
            add_tensor_node(tensor_id, size_info)
            dot.edge(tensor_str, subfunc_id)
        
        # Add edges from this subfunction to its output tensors
        for output_tensor in subfunc.output_tensor:
            tensor_id = output_tensor.tensor_id
            tensor_str = str(tensor_id)
            size_info = ""
            if output_tensor.size_h is not None and output_tensor.size_v is not None:
                size_info = f"\n{output_tensor.size_h}×{output_tensor.size_v}"
            add_tensor_node(tensor_id, size_info)
            dot.edge(subfunc_id, tensor_str)
        
        return subfunc_id
    
    # Process subfunctions by layer (m value)
    for m in sorted(subfuncs_by_m.keys()):
        # Create a subgraph cluster for this layer
        with dot.subgraph(name=f"cluster_layer_{m}") as c:
            c.attr(label=f"Layer m={m}", style="filled", color="lightgrey")
            
            # Group by operation type within this layer
            subfuncs_by_op = {}
            for subfunc in subfuncs_by_m[m]:
                if subfunc.op_type not in subfuncs_by_op:
                    subfuncs_by_op[subfunc.op_type] = []
                subfuncs_by_op[subfunc.op_type].append(subfunc)
            
            # Process each operation type
            for op_type, subfuncs in subfuncs_by_op.items():
                # Create a subgraph for this operation type
                with c.subgraph(name=f"cluster_{op_type.value}_{m}") as c2:
                    c2.attr(label=f"{op_type.value}", style="filled", 
                           color=op_colors.get(op_type, "white"), fillcolor=op_colors.get(op_type, "white"))
                    
                    # Special handling for MVM operations - arrange in a grid
                    if op_type == OperationType.MVM:
                        # Group by k
                        subfuncs_by_k = {}
                        for subfunc in subfuncs:
                            if subfunc.k not in subfuncs_by_k:
                                subfuncs_by_k[subfunc.k] = {}
                            subfuncs_by_k[subfunc.k][(subfunc.i, subfunc.j)] = subfunc
                        
                        # Process each k group
                        for k, subfuncs_by_ij in subfuncs_by_k.items():
                            with c2.subgraph(name=f"cluster_mvm_{k}_{m}") as c3:
                                c3.attr(label=f"k={k}", style="filled", color="lightgrey")
                                
                                # Add MVM subfunctions in a grid
                                for (i, j), subfunc in subfuncs_by_ij.items():
                                    add_subfunction_node(subfunc, c3)
                    else:
                        # Add other subfunctions
                        for subfunc in subfuncs:
                            add_subfunction_node(subfunc, c2)
    
    # Render the graph
    dot.render(filename, view=True)
    print(f"Graph saved as {filename}.pdf")

def visualize_compiled_model_simple(compiled_model: CompiledModel, filename: str = "compiled_model_simple"):
    """
    Create a simplified visualization of the compiled model focusing on dataflow
    
    Args:
        compiled_model: The compiled model to visualize
        filename: Base filename for the output (without extension)
    """
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Compiled Model (Simple)', format='pdf')
    dot.attr(rankdir='TB', size='11,8', ratio='fill', splines='ortho')
    
    # Track tensors that have been added to the graph
    added_tensors = set()
    
    # Group subfunctions by m (sequence order) and k (parallel function)
    subfuncs_by_mk = {}
    for subfunc in compiled_model.subfunctions:
        key = (subfunc.m, subfunc.k)
        if key not in subfuncs_by_mk:
            subfuncs_by_mk[key] = []
        subfuncs_by_mk[key].append(subfunc)
    
    # Define colors for different operation types
    op_colors = {
        OperationType.MVM: "lightblue",
        OperationType.ACTIVATION: "lightgreen",
        OperationType.TRIVIAL_COPY: "lightyellow",
        OperationType.DOT_PRODUCT: "lightpink",
        OperationType.GLU: "orange",
        OperationType.ADD: "salmon",
        OperationType.CONCAT: "violet",
        OperationType.DISTRIBUTE: "cyan",
        OperationType.PASS: "gray"
    }
    
    # Helper function to create a short tensor label
    def get_short_tensor_label(tensor_id):
        if tensor_id.i is not None and tensor_id.j is not None:
            return f"T{tensor_id.k}.{tensor_id.m}({tensor_id.i},{tensor_id.j})"
        else:
            return f"T{tensor_id.k}.{tensor_id.m}"
    
    # Helper function to add a tensor node if it hasn't been added yet
    def add_tensor_node(tensor_id, size_info=""):
        tensor_str = str(tensor_id)  # Use full string as node ID
        short_label = get_short_tensor_label(tensor_id)
        
        if tensor_id not in added_tensors:
            label = f"{short_label}{size_info}"
            dot.node(tensor_str, label, shape="ellipse", style="filled", fillcolor="lightgray")
            added_tensors.add(tensor_id)
    
    # Helper function to add a subfunction node with simplified label
    def add_subfunction_node(subfunc):
        subfunc_id = f"func_{subfunc.i}_{subfunc.j}_{subfunc.k}_{subfunc.m}_{subfunc.n}_{subfunc.op_type.value}"
        
        # Create shorter label for subfunctions
        op_abbr = ''.join([c for c in subfunc.op_type.value if c.isupper()])
        if not op_abbr:
            op_abbr = subfunc.op_type.value[:3]
            
        if subfunc.op_type == OperationType.MVM:
            label = f"{op_abbr}({subfunc.i},{subfunc.j})"
        else:
            label = f"{op_abbr}"
            if subfunc.i != 0 or subfunc.j != 0:
                label += f"({subfunc.i},{subfunc.j})"
        
        dot.node(subfunc_id, label, shape="box", style="filled", 
                 fillcolor=op_colors.get(subfunc.op_type, "white"))
        
        # Add edges from input tensors to this subfunction
        for input_tensor in subfunc.input_tensors:
            tensor_id = input_tensor.tensor_id
            tensor_str = str(tensor_id)
            add_tensor_node(tensor_id)
            dot.edge(tensor_str, subfunc_id)
        
        # Add edges from this subfunction to its output tensors
        for output_tensor in subfunc.output_tensor:
            tensor_id = output_tensor.tensor_id
            tensor_str = str(tensor_id)
            add_tensor_node(tensor_id)
            dot.edge(subfunc_id, tensor_str)
        
        return subfunc_id
    
    # Process subfunctions by layer (m value)
    for m in sorted(set(key[0] for key in subfuncs_by_mk.keys())):
        # Create a subgraph cluster for this layer
        with dot.subgraph(name=f"cluster_layer_{m}") as c:
            c.attr(label=f"Layer m={m}", style="filled", color="lightgrey")
            
            # Get all k values for this m
            k_values = sorted(set(key[1] for key in subfuncs_by_mk.keys() if key[0] == m))
            
            # Process each parallel function
            for k in k_values:
                # Create a subgraph for this parallel function
                with c.subgraph(name=f"cluster_func_{m}_{k}") as c2:
                    c2.attr(label=f"k={k}", style="filled", color="lightgrey")
                    
                    # Get subfunctions for this (m,k)
                    subfuncs = subfuncs_by_mk.get((m, k), [])
                    
                    # Group by operation type
                    subfuncs_by_op = {}
                    for subfunc in subfuncs:
                        if subfunc.op_type not in subfuncs_by_op:
                            subfuncs_by_op[subfunc.op_type] = []
                        subfuncs_by_op[subfunc.op_type].append(subfunc)
                    
                    # Process each operation type
                    for op_type in sorted(subfuncs_by_op.keys(), key=lambda x: x.value):
                        op_subfuncs = subfuncs_by_op[op_type]
                        
                        # Add subfunctions
                        for subfunc in op_subfuncs:
                            add_subfunction_node(subfunc)
    
    # Render the graph
    dot.render(filename, view=True)
    print(f"Graph saved as {filename}.pdf")

#=======================model_save=========================
def parse_compute_graph(compiled_model: CompiledModel) -> Dict:
    """
    Parse the compiled model to extract connection information for hardware mapping
    
    Args:
        compiled_model: The compiled model with subfunctions
        
    Returns:
        Dictionary containing connection information and data flow details
    """
    # Build dependency graph if not already built
    if not compiled_model.dependency_graph:
        compiled_model.build_dependency_graph()
    
    # Create a dictionary to store connection information
    connection_info = {
        'nodes': [],               # List of all computation nodes (subfunctions)
        'tensors': set(),          # Set of all tensors
        'tensor_producers': {},    # Map from tensor ID to producing subfunction
        'tensor_consumers': {},    # Map from tensor ID to consuming subfunctions
        'subfunction_inputs': {},  # Map from subfunction to input tensors
        'subfunction_outputs': {}, # Map from subfunction to output tensors
        'operation_groups': {},    # Group subfunctions by operation type
        'sequential_stages': {},   # Group subfunctions by sequential stage (m)
        'parallel_groups': {},     # Group subfunctions by parallel index (k)
        'spatial_mapping': {}      # Spatial organization (i,j coordinates)
    }
    
    # Extract nodes (subfunctions)
    for idx, subfunc in enumerate(compiled_model.subfunctions):
        # Assign a unique ID to each subfunction
        subfunction_id = f"sf_{subfunc.k}_{subfunc.m}_{subfunc.n}_{subfunc.i}_{subfunc.j}"
        
        # Create node entry
        node = {
            'id': subfunction_id,
            'op_type': subfunc.op_type.value,
            'coordinates': (subfunc.i, subfunc.j, subfunc.k, subfunc.m, subfunc.n),
            'shape': subfunc.shape
        }
        connection_info['nodes'].append(node)
        
        # Track inputs and outputs for this subfunction
        input_tensors = []
        for input_tensor in subfunc.input_tensors:
            tensor_id = (input_tensor.tensor_id.i, input_tensor.tensor_id.j, 
                         input_tensor.tensor_id.k, input_tensor.tensor_id.m, 
                         input_tensor.tensor_id.n)
            
            # Add to tensors set
            connection_info['tensors'].add(tensor_id)
            
            # Add to tensor consumers
            if tensor_id not in connection_info['tensor_consumers']:
                connection_info['tensor_consumers'][tensor_id] = []
            connection_info['tensor_consumers'][tensor_id].append(subfunction_id)
            
            # Add tensor details
            input_tensors.append({
                'tensor_id': tensor_id,
                'size_h': input_tensor.size_h,
                'size_v': input_tensor.size_v
            })
        
        output_tensors = []
        for output_tensor in subfunc.output_tensor:
            tensor_id = (output_tensor.tensor_id.i, output_tensor.tensor_id.j, 
                         output_tensor.tensor_id.k, output_tensor.tensor_id.m, 
                         output_tensor.tensor_id.n)
            
            # Add to tensors set
            connection_info['tensors'].add(tensor_id)
            
            # Set tensor producer
            connection_info['tensor_producers'][tensor_id] = subfunction_id
            
            # Add tensor details
            output_tensors.append({
                'tensor_id': tensor_id,
                'size_h': output_tensor.size_h,
                'size_v': output_tensor.size_v
            })
        
        # Store subfunction I/O
        connection_info['subfunction_inputs'][subfunction_id] = input_tensors
        connection_info['subfunction_outputs'][subfunction_id] = output_tensors
        
        # Group by operation type
        op_type = subfunc.op_type.value
        if op_type not in connection_info['operation_groups']:
            connection_info['operation_groups'][op_type] = []
        connection_info['operation_groups'][op_type].append(subfunction_id)
        
        # Group by sequential stage (m)
        stage_key = f"stage_{subfunc.m}"
        if stage_key not in connection_info['sequential_stages']:
            connection_info['sequential_stages'][stage_key] = []
        connection_info['sequential_stages'][stage_key].append(subfunction_id)
        
        # Group by parallel index (k)
        parallel_key = f"parallel_{subfunc.k}"
        if parallel_key not in connection_info['parallel_groups']:
            connection_info['parallel_groups'][parallel_key] = []
        connection_info['parallel_groups'][parallel_key].append(subfunction_id)
        
        # Track spatial mapping
        spatial_key = (subfunc.i, subfunc.j)
        if spatial_key not in connection_info['spatial_mapping']:
            connection_info['spatial_mapping'][spatial_key] = []
        connection_info['spatial_mapping'][spatial_key].append(subfunction_id)
    
    # Extract data flow paths
    connection_info['data_flow_paths'] = extract_data_flow_paths(connection_info)
    
    return connection_info

def extract_data_flow_paths(connection_info: Dict) -> List[Dict]:
    """
    Extract data flow paths from connection information
    
    Args:
        connection_info: Connection information dictionary
        
    Returns:
        List of data flow paths
    """
    paths = []
    
    # Find all source nodes (nodes with no inputs or only inputs from outside the model)
    source_nodes = []
    for node in connection_info['nodes']:
        node_id = node['id']
        has_internal_inputs = False
        
        for input_tensor in connection_info['subfunction_inputs'][node_id]:
            tensor_id = input_tensor['tensor_id']
            if tensor_id in connection_info['tensor_producers']:
                has_internal_inputs = True
                break
        
        if not has_internal_inputs:
            source_nodes.append(node_id)
    
    # For each source node, trace all possible paths
    for source_node in source_nodes:
        trace_paths(source_node, [], connection_info, paths)
    
    return paths

def trace_paths(current_node: str, current_path: List[str], 
                connection_info: Dict, all_paths: List[Dict]):
    """
    Recursively trace all paths from a given node
    
    Args:
        current_node: Current node ID
        current_path: Path traversed so far
        connection_info: Connection information dictionary
        all_paths: List to store all found paths
    """
    # Add current node to path
    path = current_path + [current_node]
    
    # Find all output tensors from this node
    output_tensors = []
    for output_tensor in connection_info['subfunction_outputs'][current_node]:
        tensor_id = output_tensor['tensor_id']
        output_tensors.append(tensor_id)
    
    # Find all nodes that consume these output tensors
    next_nodes = set()
    for tensor_id in output_tensors:
        if tensor_id in connection_info['tensor_consumers']:
            for consumer in connection_info['tensor_consumers'][tensor_id]:
                if consumer not in path:  # Avoid cycles
                    next_nodes.add(consumer)
    
    if not next_nodes:
        # This is a sink node, add the complete path
        node_details = []
        tensor_transfers = []
        
        # Add node details
        for node_id in path:
            for node in connection_info['nodes']:
                if node['id'] == node_id:
                    node_details.append(node)
                    break
        
        # Add tensor transfers between nodes
        for i in range(len(path) - 1):
            producer = path[i]
            consumer = path[i + 1]
            
            # Find tensors that flow from producer to consumer
            for output_tensor in connection_info['subfunction_outputs'][producer]:
                tensor_id = output_tensor['tensor_id']
                if tensor_id in connection_info['tensor_consumers'] and consumer in connection_info['tensor_consumers'][tensor_id]:
                    tensor_transfers.append({
                        'from': producer,
                        'to': consumer,
                        'tensor_id': tensor_id,
                        'size_h': output_tensor['size_h'],
                        'size_v': output_tensor['size_v']
                    })
        
        all_paths.append({
            'nodes': node_details,
            'transfers': tensor_transfers
        })
    else:
        # Continue tracing paths
        for next_node in next_nodes:
            trace_paths(next_node, path, connection_info, all_paths)

def save_compute_graph(connection_info: Dict, filename: str):
    """
    Save the compute graph connection information to a file
    
    Args:
        connection_info: Connection information dictionary
        filename: Output filename
    """
    import json
    
    # Convert sets to lists for JSON serialization
    serializable_info = connection_info.copy()
    serializable_info['tensors'] = [list(tensor_id) for tensor_id in connection_info['tensors']]
    
    # Convert tuple keys to string keys for JSON serialization
    tensor_producers = {}
    for tensor_id, producer in connection_info['tensor_producers'].items():
        tensor_producers[str(tensor_id)] = producer
    serializable_info['tensor_producers'] = tensor_producers
    
    tensor_consumers = {}
    for tensor_id, consumers in connection_info['tensor_consumers'].items():
        tensor_consumers[str(tensor_id)] = consumers
    serializable_info['tensor_consumers'] = tensor_consumers
    
    spatial_mapping = {}
    for coords, nodes in connection_info['spatial_mapping'].items():
        spatial_mapping[str(coords)] = nodes
    serializable_info['spatial_mapping'] = spatial_mapping
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(serializable_info, f, indent=2)


def visualize_compute_graph_graphviz(connection_info: Dict, output_file: str = 'compute_graph'):
    """
    Visualize the compute graph using Graphviz
    
    Args:
        connection_info: Connection information dictionary
        output_file: Output filename (without extension)
    """
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Compute Graph', format='pdf')
    dot.attr(rankdir='TB', size='11,8', ratio='fill')
    
    # Define colors for different operation types
    op_colors = {
        'MVM': 'lightblue',
        'ACTIVATION': 'lightgreen',
        'TRIVIAL_COPY': 'lightgray',
        'GLU': 'lightyellow',
        'ADD': 'lightcoral',
        'CONCAT': 'lightpink',
        'DISTRIBUTE': 'lavender',
        'PASS': 'white'
    }
    
    # Group nodes by sequential stage (m)
    nodes_by_stage = {}
    for node in connection_info['nodes']:
        m = node['coordinates'][3]  # m coordinate
        if m not in nodes_by_stage:
            nodes_by_stage[m] = []
        nodes_by_stage[m].append(node)
    
    # Add nodes to the graph, grouped by stage
    for stage, nodes in sorted(nodes_by_stage.items()):
        # Create a subgraph for this stage
        with dot.subgraph(name=f'cluster_stage_{stage}') as c:
            c.attr(label=f'Stage {stage}', style='filled', color='lightgray')
            
            # Group nodes by operation type within this stage
            nodes_by_op = {}
            for node in nodes:
                op_type = node['op_type']
                if op_type not in nodes_by_op:
                    nodes_by_op[op_type] = []
                nodes_by_op[op_type].append(node)
            
            # Add nodes for each operation type
            for op_type, op_nodes in sorted(nodes_by_op.items()):
                # Create a subgraph for this operation type
                with c.subgraph(name=f'cluster_{op_type}_{stage}') as c2:
                    c2.attr(label=f'{op_type}', style='filled', color=op_colors.get(op_type, 'white'))
                    
                    # Add nodes
                    for node in op_nodes:
                        node_id = node['id']
                        i, j = node['coordinates'][0], node['coordinates'][1]
                        label = f"{op_type}\n({i},{j})"
                        
                        if node['shape']:
                            label += f"\n{node['shape'][0]}×{node['shape'][1]}"
                        
                        c2.node(node_id, label, shape='box', style='filled', 
                               fillcolor=op_colors.get(op_type, 'white'))
    
    # Add edges
    for tensor_id, consumers in connection_info['tensor_consumers'].items():
        producer = connection_info['tensor_producers'].get(tensor_id)
        if producer:
            for consumer in consumers:
                # Find tensor size for edge label
                size_info = ""
                for output_tensor in connection_info['subfunction_outputs'][producer]:
                    if output_tensor['tensor_id'] == tensor_id:
                        h, v = output_tensor['size_h'], output_tensor['size_v']
                        if h is not None and v is not None:
                            size_info = f"{h}×{v}"
                            break
                
                dot.edge(producer, consumer, label=size_info)
    
    # Render the graph
    dot.render(output_file, view=True)
    print(f"Graph visualization saved to {output_file}.pdf")

def analyze_compute_graph(connection_info: Dict) -> Dict:
    """
    Analyze the compute graph to extract useful statistics and insights
    
    Args:
        connection_info: Connection information dictionary
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'node_count': len(connection_info['nodes']),
        'tensor_count': len(connection_info['tensors']),
        'op_type_counts': {},
        'data_transfer_volume': 0,
        'critical_path_length': 0,
        'max_fanout': 0,
        'max_fanin': 0
    }
    
    # Count operation types
    for node in connection_info['nodes']:
        op_type = node['op_type']
        if op_type not in analysis['op_type_counts']:
            analysis['op_type_counts'][op_type] = 0
        analysis['op_type_counts'][op_type] += 1
    
    # Calculate data transfer volume
    for transfers in connection_info['data_flow_paths']:
        for transfer in transfers.get('transfers', []):
            size_h = transfer.get('size_h', 1)
            size_v = transfer.get('size_v', 1)
            if size_h is not None and size_v is not None:
                analysis['data_transfer_volume'] += size_h * size_v
    
    # Find critical path length
    max_path_length = 0
    for path in connection_info['data_flow_paths']:
        path_length = len(path.get('nodes', []))
        max_path_length = max(max_path_length, path_length)
    analysis['critical_path_length'] = max_path_length
    
    # Find max fanout (number of consumers for a tensor)
    for tensor_id, consumers in connection_info['tensor_consumers'].items():
        analysis['max_fanout'] = max(analysis['max_fanout'], len(consumers))
    
    # Find max fanin (number of input tensors for a node)
    for node_id, inputs in connection_info['subfunction_inputs'].items():
        analysis['max_fanin'] = max(analysis['max_fanin'], len(inputs))
    
    return analysis