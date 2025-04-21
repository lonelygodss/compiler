#=======================visualization=========================
import graphviz
from typing import Dict, Set, List, Tuple
import math

def visualize_compiled_model(compiled_model: CompiledModel, output_file: str = "compiled_model"):
    """
    Visualize the compiled model as a compute graph
    
    Args:
        compiled_model: The compiled model to visualize
        output_file: Base name for the output file (without extension)
    
    Returns:
        None (saves the visualization to a file)
    """
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Compiled Model', format='pdf')
    dot.attr(rankdir='TB', size='11,8', dpi='300')
    
    # Define node styles
    dot.attr('node', shape='box', style='filled', fontname='Arial', fontsize='10')
    
    # Group subfunctions by their parent function (k, m, n)
    subfuncs_by_parent = {}
    for subfunc in compiled_model.subfunctions:
        key = (subfunc.k, subfunc.m, subfunc.n)
        if key not in subfuncs_by_parent:
            subfuncs_by_parent[key] = []
        subfuncs_by_parent[key].append(subfunc)
    
    # Track tensors we've already added
    added_tensors = set()
    
    # Track subfunctions we've already added
    added_subfuncs = set()
    
    # Process each parent function group
    for (k, m, n), subfuncs in sorted(subfuncs_by_parent.items()):
        # Create a subgraph for this function group
        with dot.subgraph(name=f'cluster_{k}_{m}_{n}') as c:
            c.attr(label=f'Function(k={k}, m={m}, n={n})', style='rounded', color='blue', fontcolor='blue')
            
            # For MVM functions, organize subfunctions in a grid
            mvm_subfuncs = [sf for sf in subfuncs if sf.op_type == OperationType.MVM]
            if mvm_subfuncs:
                # Find the max i and j values to determine grid dimensions
                max_i = max([sf.i for sf in mvm_subfuncs])
                max_j = max([sf.j for sf in mvm_subfuncs])
                
                # Create a subgraph for the MVM grid
                with c.subgraph(name=f'cluster_mvm_{k}_{m}_{n}') as mvm_grid:
                    mvm_grid.attr(label=f'MVM Matrix', style='rounded', color='darkgreen')
                    
                    # Create a grid layout
                    for j in range(max_j + 1):  # j is row (vertical)
                        for i in range(max_i + 1):  # i is column (horizontal)
                            # Find the subfunction at this position
                            sf = next((sf for sf in mvm_subfuncs if sf.i == i and sf.j == j), None)
                            if sf:
                                # Add the subfunction node
                                node_id = f'subfunc_{sf.i}_{sf.j}_{sf.k}_{sf.m}_{sf.n}'
                                mvm_grid.node(node_id, 
                                             label=f'MVM[{sf.i},{sf.j}]\n({sf.shape[0]}×{sf.shape[1]})', 
                                             fillcolor='lightgreen')
                                added_subfuncs.add(sf)
                                
                                # Add the output tensor node
                                if sf.output_tensor_id:
                                    tensor_id = f'tensor_{sf.output_tensor_id.f}_{sf.output_tensor_id.i}_{sf.output_tensor_id.j}_{sf.output_tensor_id.k}_{sf.output_tensor_id.m}_{sf.output_tensor_id.n}'
                                    if tensor_id not in added_tensors:
                                        mvm_grid.node(tensor_id, 
                                                     label=f'T[{sf.output_tensor_id.i},{sf.output_tensor_id.j}]', 
                                                     shape='ellipse', 
                                                     fillcolor='lightyellow')
                                        added_tensors.add(tensor_id)
                                    
                                    # Connect subfunction to its output
                                    mvm_grid.edge(node_id, tensor_id)
            
            # Add other subfunctions (ADD, CONCAT, etc.)
            other_subfuncs = [sf for sf in subfuncs if sf not in added_subfuncs]
            
            # Group by operation type
            subfuncs_by_op = {}
            for sf in other_subfuncs:
                if sf.op_type not in subfuncs_by_op:
                    subfuncs_by_op[sf.op_type] = []
                subfuncs_by_op[sf.op_type].append(sf)
            
            # Process each operation type
            for op_type, op_subfuncs in subfuncs_by_op.items():
                # Create a subgraph for this operation type
                with c.subgraph(name=f'cluster_{op_type.value}_{k}_{m}_{n}') as op_cluster:
                    op_cluster.attr(label=f'{op_type.value} Operations', style='rounded', color='darkred')
                    
                    # Add nodes for each subfunction
                    for sf in op_subfuncs:
                        node_id = f'subfunc_{sf.i}_{sf.j}_{sf.k}_{sf.m}_{sf.n}'
                        
                        # Choose color based on operation type
                        color_map = {
                            OperationType.ADD: 'lightcoral',
                            OperationType.CONCAT: 'lightblue',
                            OperationType.ACTIVATION: 'lightsalmon',
                            OperationType.GLU: 'plum',
                            OperationType.TRIVIAL_COPY: 'lightgrey',
                            OperationType.DOT_PRODUCT: 'lightsteelblue'
                        }
                        fillcolor = color_map.get(sf.op_type, 'white')
                        
                        # Create label based on operation type
                        if sf.op_type == OperationType.ADD:
                            label = f'ADD[j={sf.j}]'
                        elif sf.op_type == OperationType.CONCAT:
                            label = f'CONCAT'
                        else:
                            label = f'{sf.op_type.value}[j={sf.j}]'
                        
                        op_cluster.node(node_id, label=label, fillcolor=fillcolor)
                        
                        # Add the output tensor node
                        if sf.output_tensor_id:
                            tensor_id = f'tensor_{sf.output_tensor_id.f}_{sf.output_tensor_id.i}_{sf.output_tensor_id.j}_{sf.output_tensor_id.k}_{sf.output_tensor_id.m}_{sf.output_tensor_id.n}'
                            if tensor_id not in added_tensors:
                                if sf.op_type == OperationType.CONCAT:
                                    # Final output tensor has a different style
                                    op_cluster.node(tensor_id, 
                                                  label=f'Output\nT[{k},{m},{n}]', 
                                                  shape='ellipse', 
                                                  fillcolor='gold',
                                                  penwidth='2')
                                else:
                                    op_cluster.node(tensor_id, 
                                                  label=f'T[{sf.output_tensor_id.i},{sf.output_tensor_id.j}]', 
                                                  shape='ellipse', 
                                                  fillcolor='lightyellow')
                                added_tensors.add(tensor_id)
                            
                            # Connect subfunction to its output
                            op_cluster.edge(node_id, tensor_id)
    
    # Add input tensors
    for subfunc in compiled_model.subfunctions:
        node_id = f'subfunc_{subfunc.i}_{subfunc.j}_{subfunc.k}_{subfunc.m}_{subfunc.n}'
        
        # Add edges from input tensors to this subfunction
        for input_tensor in subfunc.input_tensors:
            tensor_id = None
            
            # Handle external inputs (from previous layers)
            if input_tensor.tensor_id.m == 0 and input_tensor.tensor_id.k == 0:
                tensor_id = f'input_{input_tensor.tensor_id.n}'
                if tensor_id not in added_tensors:
                    dot.node(tensor_id, 
                            label=f'Input\nT[{input_tensor.tensor_id.k},{input_tensor.tensor_id.m},{input_tensor.tensor_id.n}]', 
                            shape='ellipse', 
                            fillcolor='lightblue',
                            penwidth='2')
                    added_tensors.add(tensor_id)
            else:
                # Handle internal tensors
                tensor_id = f'tensor_{input_tensor.tensor_id.f}_{input_tensor.tensor_id.i}_{input_tensor.tensor_id.j}_{input_tensor.tensor_id.k}_{input_tensor.tensor_id.m}_{input_tensor.tensor_id.n}'
            
            # Add the edge if the tensor exists
            if tensor_id and tensor_id in added_tensors:
                # Add slice information to edge if present
                if (input_tensor.start_h is not None and 
                    input_tensor.end_h is not None and 
                    input_tensor.start_v is not None and 
                    input_tensor.end_v is not None):
                    label = f'[{input_tensor.start_h}:{input_tensor.end_h}, {input_tensor.start_v}:{input_tensor.end_v}]'
                    dot.edge(tensor_id, node_id, label=label, fontsize='8')
                else:
                    dot.edge(tensor_id, node_id)
    
    # Render the graph
    dot.render(output_file, view=True, cleanup=True)
    print(f"Visualization saved to {output_file}.pdf")

def visualize_execution_flow(compiled_model: CompiledModel, output_file: str = "execution_flow"):
    """
    Visualize the execution flow of the compiled model
    
    Args:
        compiled_model: The compiled model to visualize
        output_file: Base name for the output file (without extension)
    
    Returns:
        None (saves the visualization to a file)
    """
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Execution Flow', format='pdf')
    dot.attr(rankdir='LR', size='11,8', dpi='300')
    
    # Define node styles
    dot.attr('node', shape='box', style='filled', fontname='Arial', fontsize='10')
    
    # Rebuild the dependency graph if needed
    if not compiled_model.dependency_graph:
        compiled_model.build_dependency_graph()
    
    # Create a topological sort of subfunctions
    visited = set()
    temp_visited = set()
    order = []
    
    def visit(subfunction):
        if subfunction in visited:
            return
        if subfunction in temp_visited:
            # This would indicate a cycle, which shouldn't happen in our DAG
            raise ValueError("Cycle detected in the dependency graph")
        
        temp_visited.add(subfunction)
        
        # Visit all dependent subfunctions
        if subfunction.output_tensor_id in compiled_model.dependency_graph:
            for dependent in compiled_model.dependency_graph[subfunction.output_tensor_id]:
                visit(dependent)
        
        temp_visited.remove(subfunction)
        visited.add(subfunction)
        order.append(subfunction)
    
    # Find all subfunctions without dependencies (starting points)
    starting_subfuncs = []
    for subfunc in compiled_model.subfunctions:
        has_internal_deps = False
        for input_tensor in subfunc.input_tensors:
            # Check if this is an internal dependency
            if not (input_tensor.tensor_id.m == 0 and input_tensor.tensor_id.k == 0):
                has_internal_deps = True
                break
        
        if not has_internal_deps:
            starting_subfuncs.append(subfunc)
    
    # Perform topological sort
    for subfunc in starting_subfuncs:
        visit(subfunc)
    
    # Reverse the order to get correct execution sequence
    order.reverse()
    
    # Group subfunctions by their parent function (k, m, n)
    subfuncs_by_parent = {}
    for subfunc in order:
        key = (subfunc.k, subfunc.m, subfunc.n)
        if key not in subfuncs_by_parent:
            subfuncs_by_parent[key] = []
        subfuncs_by_parent[key].append(subfunc)
    
    # Track tensors we've already added
    added_tensors = set()
    
    # Create "time steps" for visualization
    time_steps = []
    current_step = []
    
    for subfunc in order:
        # Check if all dependencies are satisfied
        deps_satisfied = True
        for input_tensor in subfunc.input_tensors:
            tensor_id = f'tensor_{input_tensor.tensor_id.f}_{input_tensor.tensor_id.i}_{input_tensor.tensor_id.j}_{input_tensor.tensor_id.k}_{input_tensor.tensor_id.m}_{input_tensor.tensor_id.n}'
            if tensor_id not in added_tensors and not (input_tensor.tensor_id.m == 0 and input_tensor.tensor_id.k == 0):
                deps_satisfied = False
                break
        
        if deps_satisfied:
            current_step.append(subfunc)
            # Add the output tensor
            if subfunc.output_tensor_id:
                tensor_id = f'tensor_{subfunc.output_tensor_id.f}_{subfunc.output_tensor_id.i}_{subfunc.output_tensor_id.j}_{subfunc.output_tensor_id.k}_{subfunc.output_tensor_id.m}_{subfunc.output_tensor_id.n}'
                added_tensors.add(tensor_id)
        else:
            # Start a new time step
            if current_step:
                time_steps.append(current_step)
                current_step = [subfunc]
                # Add the output tensor
                if subfunc.output_tensor_id:
                    tensor_id = f'tensor_{subfunc.output_tensor_id.f}_{subfunc.output_tensor_id.i}_{subfunc.output_tensor_id.j}_{subfunc.output_tensor_id.k}_{subfunc.output_tensor_id.m}_{subfunc.output_tensor_id.n}'
                    added_tensors.add(tensor_id)
    
    # Add the last time step if not empty
    if current_step:
        time_steps.append(current_step)
    
    # Reset for visualization
    added_tensors = set()
    
    # Create subgraphs for each time step
    for step_idx, step in enumerate(time_steps):
        with dot.subgraph(name=f'cluster_step_{step_idx}') as c:
            c.attr(label=f'Step {step_idx+1}', style='rounded', color='gray')
            
            # Group subfunctions by their parent function
            step_by_parent = {}
            for subfunc in step:
                key = (subfunc.k, subfunc.m, subfunc.n)
                if key not in step_by_parent:
                    step_by_parent[key] = []
                step_by_parent[key].append(subfunc)
            
            # Process each parent function group
            for (k, m, n), parent_subfuncs in sorted(step_by_parent.items()):
                with c.subgraph(name=f'cluster_{step_idx}_{k}_{m}_{n}') as parent_c:
                    parent_c.attr(label=f'Function(k={k}, m={m}, n={n})', style='rounded', color='blue', fontcolor='blue')
                    
                    # Add nodes for each subfunction
                    for subfunc in parent_subfuncs:
                        node_id = f'step_{step_idx}_subfunc_{subfunc.i}_{subfunc.j}_{subfunc.k}_{subfunc.m}_{subfunc.n}'
                        
                        # Choose color based on operation type
                        color_map = {
                            OperationType.MVM: 'lightgreen',
                            OperationType.ADD: 'lightcoral',
                            OperationType.CONCAT: 'lightblue',
                            OperationType.ACTIVATION: 'lightsalmon',
                            OperationType.GLU: 'plum',
                            OperationType.TRIVIAL_COPY: 'lightgrey',
                            OperationType.DOT_PRODUCT: 'lightsteelblue'
                        }
                        fillcolor = color_map.get(subfunc.op_type, 'white')
                        
                        # Create label based on operation type
                        if subfunc.op_type == OperationType.MVM:
                            label = f'MVM[{subfunc.i},{subfunc.j}]\n({subfunc.shape[0]}×{subfunc.shape[1]})'
                        elif subfunc.op_type == OperationType.ADD:
                            label = f'ADD[j={subfunc.j}]'
                        elif subfunc.op_type == OperationType.CONCAT:
                            label = f'CONCAT'
                        else:
                            label = f'{subfunc.op_type.value}[j={subfunc.j}]'
                        
                        parent_c.node(node_id, label=label, fillcolor=fillcolor)
                        
                        # Add the output tensor node
                        if subfunc.output_tensor_id:
                            tensor_id = f'step_{step_idx}_tensor_{subfunc.output_tensor_id.f}_{subfunc.output_tensor_id.i}_{subfunc.output_tensor_id.j}_{subfunc.output_tensor_id.k}_{subfunc.output_tensor_id.m}_{subfunc.output_tensor_id.n}'
                            
                            if subfunc.op_type == OperationType.CONCAT and subfunc.i == 0 and subfunc.j == 0:
                                # Final output tensor has a different style
                                parent_c.node(tensor_id, 
                                            label=f'Output\nT[{k},{m},{n}]', 
                                            shape='ellipse', 
                                            fillcolor='gold',
                                            penwidth='2')
                            else:
                                parent_c.node(tensor_id, 
                                            label=f'T[{subfunc.output_tensor_id.i},{subfunc.output_tensor_id.j}]', 
                                            shape='ellipse', 
                                            fillcolor='lightyellow')
                            
                            # Connect subfunction to its output
                            parent_c.edge(node_id, tensor_id)
                            
                            # Track this tensor for dependencies
                            global_tensor_id = f'tensor_{subfunc.output_tensor_id.f}_{subfunc.output_tensor_id.i}_{subfunc.output_tensor_id.j}_{subfunc.output_tensor_id.k}_{subfunc.output_tensor_id.m}_{subfunc.output_tensor_id.n}'
                            added_tensors.add(global_tensor_id)
    
    # Add edges between time steps
    for step_idx in range(1, len(time_steps)):
        for subfunc in time_steps[step_idx]:
            target_node_id = f'step_{step_idx}_subfunc_{subfunc.i}_{subfunc.j}_{subfunc.k}_{subfunc.m}_{subfunc.n}'
            
            # Add edges from input tensors to this subfunction
            for input_tensor in subfunc.input_tensors:
                # Skip external inputs
                if input_tensor.tensor_id.m == 0 and input_tensor.tensor_id.k == 0:
                    continue
                
                # Find the source tensor in previous steps
                for prev_step_idx in range(step_idx):
                    for prev_subfunc in time_steps[prev_step_idx]:
                        if (prev_subfunc.output_tensor_id and 
                            prev_subfunc.output_tensor_id.f == input_tensor.tensor_id.f and
                            prev_subfunc.output_tensor_id.i == input_tensor.tensor_id.i and
                            prev_subfunc.output_tensor_id.j == input_tensor.tensor_id.j and
                            prev_subfunc.output_tensor_id.k == input_tensor.tensor_id.k and
                            prev_subfunc.output_tensor_id.m == input_tensor.tensor_id.m and
                            prev_subfunc.output_tensor_id.n == input_tensor.tensor_id.n):
                            
                            source_tensor_id = f'step_{prev_step_idx}_tensor_{prev_subfunc.output_tensor_id.f}_{prev_subfunc.output_tensor_id.i}_{prev_subfunc.output_tensor_id.j}_{prev_subfunc.output_tensor_id.k}_{prev_subfunc.output_tensor_id.m}_{prev_subfunc.output_tensor_id.n}'
                            
                            # Add slice information to edge if present
                            if (input_tensor.start_h is not None and 
                                input_tensor.end_h is not None and 
                                input_tensor.start_v is not None and 
                                input_tensor.end_v is not None):
                                label = f'[{input_tensor.start_h}:{input_tensor.end_h}, {input_tensor.start_v}:{input_tensor.end_v}]'
                                dot.edge(source_tensor_id, target_node_id, label=label, fontsize='8')
                            else:
                                dot.edge(source_tensor_id, target_node_id)
    
    # Add external inputs
    for step_idx, step in enumerate(time_steps):
        for subfunc in step:
            target_node_id = f'step_{step_idx}_subfunc_{subfunc.i}_{subfunc.j}_{subfunc.k}_{subfunc.m}_{subfunc.n}'
            
            # Check for external inputs
            for input_tensor in subfunc.input_tensors:
                if input_tensor.tensor_id.m == 0 and input_tensor.tensor_id.k == 0:
                    input_id = f'input_{input_tensor.tensor_id.n}'
                    
                    # Only add the input node once
                    if input_id not in added_tensors:
                        dot.node(input_id, 
                                label=f'Input\nT[{input_tensor.tensor_id.k},{input_tensor.tensor_id.m},{input_tensor.tensor_id.n}]', 
                                shape='ellipse', 
                                fillcolor='lightblue',
                                penwidth='2')
                        added_tensors.add(input_id)
                    
                    # Add slice information to edge if present
                    if (input_tensor.start_h is not None and 
                        input_tensor.end_h is not None and 
                        input_tensor.start_v is not None and 
                        input_tensor.end_v is not None):
                        label = f'[{input_tensor.start_h}:{input_tensor.end_h}, {input_tensor.start_v}:{input_tensor.end_v}]'
                        dot.edge(input_id, target_node_id, label=label, fontsize='8')
                    else:
                        dot.edge(input_id, target_node_id)
    
    # Render the graph
    dot.render(output_file, view=True, cleanup=True)
    print(f"Execution flow visualization saved to {output_file}.pdf")