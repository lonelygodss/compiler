#%%
import compiler.model_compiler.utils as utils
from enum import Enum
from typing import List, Dict, Tuple, Set, Optional, Any

class OperationType(Enum):
    MVM = "MVM"                 # Matrix-vector multiplication
    ACTIVATION = "ACTIVATION"   # Activation function (e.g., SiLU)
    TRIVIAL_COPY = "TRIVIAL_COPY"  # Passing data without change
    DOT_PRODUCT = "DOT_PRODUCT" # Element-wise multiplication
    GLU = "GLU"                 # Gated Linear Unit
    ADD = "ADD"                 # Addition operation (for aggregation)
    CONCAT = "CONCAT"           # Concatenation operation (for aggregation)
    DISTRIBUTE = "DISTRIBUTE"   # Distributing data across concatenated dimension
    PASS = "PASS"               # From concatenated result to next step

class TensorId:
    """Unique identifier for tensors in the model"""
    def __init__(self, k: int, m: int, n: int, i: int = None, j: int = None):
        self.k = k  # parallel function index within an FFN layer
        self.m = m  # sequence order within an FFN layer
        self.n = n  # decoder layer index
        self.i = i  # horizontal index (for division)
        self.j = j  # vertical index (for division)
    
    def __str__(self):
        if self.i is not None and self.j is not None:
            return f"Tensor(i={self.i}, j={self.j}, k={self.k}, m={self.m}, n={self.n})"
        else:
            return f"Tensor(k={self.k}, m={self.m}, n={self.n})"
    
    def __eq__(self, other):
        if not isinstance(other, TensorId):
            return False
        if self.i is None and self.j is None:
            return (self.k == other.k and 
                    self.m == other.m and self.n == other.n)
        return (self.i == other.i and self.j == other.j and 
                self.k == other.k and self.m == other.m and self.n == other.n)
    
    def __hash__(self):
        if self.i is None and self.j is None:
            return hash(( self.k, self.m, self.n))
        return hash((self.i, self.j, self.k, self.m, self.n))

class TensorWithSize:
    """Represents a instance of a tensor """
    def __init__(self, tensor_id: TensorId, size_h: int = None, 
                 size_v: int = None):
        self.tensor_id = tensor_id
        self.size_h = size_h  # horizontal size
        self.size_v = size_v  # vertical size
    
    def __size__(self):
        slice_info = ""
        if self.size_h is not None:
            slice_info = f", size=[{self.size_h},{self.size_v}]"
        return f"{self.tensor_id}{slice_info}"

class Function:
    """High-level function in the FFN layer (before division)"""
    def __init__(self, k: int, m: int, n: int, op_type: OperationType):
        self.k = k  # parallel function index
        self.m = m  # sequence order in FFN
        self.n = n  # decoder layer index
        self.op_type = op_type
        self.shape = None  # For MVM operations: (input_dim, output_dim)
    
    def set_shape(self, shape: Tuple[int, int]):
        """Set the shape for MVM operations (input_dim, output_dim)"""
        self.shape = shape
        
    def __str__(self):
        shape_info = f", shape={self.shape}" if self.shape else ""
        return f"Function(k={self.k}, m={self.m}, n={self.n}, op={self.op_type.value}{shape_info})"
#%%
class SubFunction:
    """Sub-function after division according to hardware constraints"""
    def __init__(self, i: int, j: int, k: int, m: int, n: int, op_type: OperationType):
        self.i = i  # horizontal index
        self.j = j  # vertical index
        self.k = k  # parallel function index
        self.m = m  # sequence order in FFN
        self.n = n  # decoder layer index
        self.op_type = op_type
        self.input_tensors: List[TensorWithSize] = []
        self.output_tensor: List[TensorWithSize] = []
        self.shape = None  # For MVM operations: (submatrix input_dim, submatrix output_dim)
        self.parent_function = None  # Reference to the parent Function
        
    def set_parent(self, parent: Function):
        """Set the parent function this subfunction was derived from"""
        self.parent_function = parent
        
    def set_shape(self, shape: Tuple[int, int]):
        """Set the shape for MVM operations (input_dim, output_dim) of the submatrix"""
        self.shape = shape
        
    def add_input_tensor(self, tensor_id: TensorId, size_h=None, size_v=None):
        """Add an input tensor dependency"""
        self.input_tensors.append(TensorWithSize(tensor_id, size_h, size_v))
        
    def add_output_tensor(self, tensor_id: TensorId, size_h=None, size_v=None):
        """Register output tensor with the function coordinates"""
        self.output_tensor.append(TensorWithSize(tensor_id, size_h, size_v))
        
    def __str__(self):
        inputs = ', '.join(str(ts.tensor_id) for ts in self.input_tensors)
        output = ', '.join(str(ts.tensor_id) for ts in self.output_tensor)
        shape_info = f", shape={self.shape}" if self.shape else ""
        return f"SubFunction(i={self.i}, j={self.j}, k={self.k}, m={self.m}, n={self.n}, op={self.op_type.value}, inputs=[{inputs}], output={output}{shape_info})"

class Model:
    """Holds the complete model description"""
    def __init__(self):
        self.functions: List[Function] = []
        
    def add_function(self, function: Function):
        """Add a function to the model"""
        self.functions.append(function)
        
    def get_function(self, k: int, m: int, n: int) -> Optional[Function]:
        """Get a function by its coordinates"""
        for func in self.functions:
            if func.k == k and func.m == m and func.n == n:
                return func
        return None
    
    def __str__(self):
        return '\n'.join(str(func) for func in self.functions)

class CompiledModel:
    """Holds the divided model with subfunctions"""
    def __init__(self):
        self.subfunctions: List[SubFunction] = []
        self.dependency_graph: Dict[TensorId, List[SubFunction]] = {}
        
    def add_subfunction(self, subfunction: SubFunction):
        """Add a subfunction to the compiled model"""
        self.subfunctions.append(subfunction)
        
        # # Update dependency graph
        # if subfunction.output_tensor:
        #     if subfunction.output_tensor.tensor_id not in self.dependency_graph:
        #         self.dependency_graph[subfunction.output_tensor.tensor_id] = []
    
    def build_dependency_graph(self):
        """Build a graph showing which subfunctions depend on which tensors"""
        # Reset the graph
        self.dependency_graph = {}
        
        # Add all output tensors to the graph
        for subfunc in self.subfunctions:
            if subfunc.output_tensor:
                for output_tensor in subfunc.output_tensor:
                    if output_tensor.tensor_id not in self.dependency_graph:
                        self.dependency_graph[output_tensor.tensor_id] = []
        
        # For each subfunction, add its dependencies
        for subfunc in self.subfunctions:
            for input_tensor in subfunc.input_tensors:
                if input_tensor.tensor_id in self.dependency_graph:
                    self.dependency_graph[input_tensor.tensor_id].append(subfunc)
        
    def __str__(self):
        return '\n'.join(str(subfunc) for subfunc in self.subfunctions)

class Compiler:
    """Compiler that divides the model according to hardware constraints"""
    def __init__(self, array_h: int, array_v: int):
        """
        Initialize compiler with array dimensions
        
        Args:
            array_h: Horizontal size of CIM array
            array_v: Vertical size of CIM array
        """
        self.array_h = array_h
        self.array_v = array_v
        
    def divide_model(self, model: Model) -> CompiledModel:
        """
        Divide the model according to hardware constraints
        
        Args:
            model: High-level model description
            
        Returns:
            Compiled model with subfunctions
        """
        compiled_model = CompiledModel()
        
        # Process each function in the model
        for function in model.functions:
            if function.op_type == OperationType.MVM:
                self._divide_mvm(function, compiled_model)
            elif function.op_type in [OperationType.ACTIVATION, OperationType.TRIVIAL_COPY, 
                                     OperationType.DOT_PRODUCT, OperationType.GLU]:
                self._divide_elementwise(function, compiled_model)
        
        
        # Build the dependency graph
        compiled_model.build_dependency_graph()
        
        return compiled_model
    
    def _divide_mvm(self, function: Function, compiled_model: CompiledModel):
        """Divide an MVM function into subfunctions based on array size constraints"""
        if not function.shape:
            raise ValueError(f"Function {function} has no shape defined, required for MVM division")
            
        input_dim, output_dim = function.shape
        
        # Calculate number of divisions needed
        h_divisions = (output_dim + self.array_h - 1) // self.array_h
        v_divisions = (input_dim + self.array_v - 1) // self.array_v
        

        # For each division, create a subfunction
        for i in range(v_divisions):
            for j in range(h_divisions):
                # Calculate the actual dimensions of this submatrix
                start_h = j * self.array_h
                end_h = min((j + 1) * self.array_h, output_dim)
                start_v = i * self.array_v
                end_v = min((i + 1) * self.array_v, input_dim)
                
                # Create subfunction
                subfunc = SubFunction(i+1, j+1, function.k, function.m, function.n, OperationType.MVM)
                subfunc.set_parent(function)
                subfunc.set_shape((end_v - start_v, end_h - start_h))
                
                # Set input tensors - for MVM, we need the appropriate vertical slice of each input
                subfunc.add_input_tensor(
                    tensor_id=TensorId(i = -(i+1), j = -(j+1), 
                                        k=function.k,
                                        m=function.m,
                                        n=function.n), 
                    size_v=1,
                    size_h=end_v - start_v
                )
                
                # Set output tensor
                subfunc.add_output_tensor(
                    tensor_id=TensorId(i = i+1, j = j+1,
                                       k=function.k,
                                       m=function.m,
                                       n=function.n),
                    size_v=1,
                    size_h=end_h - start_h
                )

                # Add to compiled model
                compiled_model.add_subfunction(subfunc)
        # Add distribution functions to handle the divided computations
        distribution_func = SubFunction(0, -1, function.k, function.m, function.n, OperationType.DISTRIBUTE)
        # Add input tensors for distribution
        distribution_func.add_input_tensor(
            tensor_id=TensorId(k=1,
                               m=function.m,
                               n=function.n), 
            size_h=input_dim,
            size_v=1
        )
        # Add output tensor to the distribution function
        for i in range(v_divisions):
            for j in range(h_divisions):
                start_v = i * self.array_v
                end_v = min((i + 1) * self.array_v, input_dim)
                start_h = j * self.array_h
                end_h = min((j + 1) * self.array_h, output_dim)
                distribution_func.add_output_tensor(
                    tensor_id=TensorId(i = -(i+1), j = -(j+1),
                                       k=function.k,
                                       m=function.m,
                                       n=function.n),
                    size_h=end_h - start_h,
                    size_v=1
                )
        # Add the distribution function to the compiled model
        compiled_model.add_subfunction(distribution_func)

        # Add concatenation function to handle the divided computations
        concat_func = SubFunction(0, 0, function.k, function.m, function.n, OperationType.CONCAT)

        for j in range(h_divisions):
            start_h = j * self.array_h
            end_h = min((j + 1) * self.array_h, output_dim)
            # Add addition function to combine horizontal slices
            add_func = SubFunction(0, j+1, function.k, function.m, function.n, OperationType.ADD)
            # Add output tensor for the addition function
            add_func.add_output_tensor(
                tensor_id=TensorId(i = 0, j = j+1,
                                   k=function.k,
                                   m=function.m,
                                   n=function.n),
                size_h=end_h - start_h,
                size_v=1
            )
            add_func.set_shape((1, end_h - start_h))  # Set shape for addition operations
            # Add input tensors to concat function
            concat_func.add_input_tensor(
                tensor_id=TensorId(i = 0, j = j+1,
                                   k=function.k,
                                   m=function.m,
                                   n=function.n),
                size_h=end_h - start_h,
                size_v=1
            )

            # Add input tensors for the addition function
            
            for i in range(v_divisions):
                start_v = i * self.array_v
                end_v = min((i + 1) * self.array_v, input_dim)
                add_func.add_input_tensor(
                    tensor_id=TensorId(i = i+1, j = j+1,
                                       k=function.k,
                                       m=function.m,
                                       n=function.n),
                    size_h=end_h - start_h,
                    size_v=1
                )
            # Add the addition function to the compiled model
            compiled_model.add_subfunction(add_func)
        # Set output tensor for the concatenation function
        concat_func.add_output_tensor(
            tensor_id=TensorId(i = 0, j = 0,
                               k=function.k,
                               m=function.m,
                               n=function.n),
            size_h=output_dim,
            size_v=1
        )
        # Add the concatenation function to the compiled model
        compiled_model.add_subfunction(concat_func)
        # Add pass function to pass to next step
        pass_func = SubFunction(-1, 0, function.k, function.m, function.n, OperationType.PASS)
        # Add input tensor for the pass function
        pass_func.add_input_tensor(
            tensor_id=TensorId(i = 0, j = 0,
                               k=function.k,
                               m=function.m,
                               n=function.n),
            size_h=output_dim,
            size_v=1
        )
        # Add output tensor for the pass function
        pass_func.add_output_tensor(
            tensor_id=TensorId(k=function.k,
                                    m=function.m+1,
                                    n=function.n),
                                    size_h=output_dim,
                                    size_v=1)
        # Add the pass function to the compiled model
        compiled_model.add_subfunction(pass_func)



    
    def _divide_elementwise(self, function: Function, compiled_model: CompiledModel):
        """Divide element-wise operations (activation, GLU, etc.) into subfunctions"""
        # For element-wise operations, we need to know the output dimension
        # We can infer this from the input tensors or from the function shape
        output_dim = None
        
        if function.shape:
            _, output_dim = function.shape
        else:
            # Try to infer from the first input tensor
            # This is a simplification - in reality, we'd need to know the exact output shape
            output_dim = 1024  # Default value, should be determined from model architecture
        
        # Calculate number of divisions needed (only in vertical dimension)
        h_divisions = (output_dim + self.array_h - 1) // self.array_h

        # Create distribution function to handle the divided computations
        if function.op_type == OperationType.GLU:
            distribution_func1 = SubFunction(0, -1, 1, function.m, function.n, OperationType.DISTRIBUTE)
            # Add input tensors for distribution
            distribution_func1.add_input_tensor(
                tensor_id=TensorId(k=1,
                                   m=function.m,
                                   n=function.n), 
                size_h=output_dim,
                size_v=1
            )
            distribution_func2 = SubFunction(0, -1, 2, function.m, function.n, OperationType.DISTRIBUTE)
            # Add input tensors for distribution
            distribution_func2.add_input_tensor(
                tensor_id=TensorId(k=2,
                                   m=function.m,
                                   n=function.n), 
                size_h=output_dim,
                size_v=1
            )
        else:
            distribution_func = SubFunction(0, -1, function.k, function.m, function.n, OperationType.DISTRIBUTE)
            # Add input tensors for distribution
            distribution_func.add_input_tensor(
                tensor_id=TensorId(k=function.k,
                                   m=function.m,
                                   n=function.n), 
                size_h=output_dim,
                size_v=1
            )
        # Add concatenation function to handle the divided computations
        concat_func = SubFunction(0, 0, function.k, function.m, function.n, OperationType.CONCAT)
        # Add output tensor for the concatenation function
        concat_func.add_output_tensor(
            tensor_id=TensorId(i = 0, j = 0,
                               k=function.k,
                               m=function.m,
                               n=function.n),
            size_h=output_dim,
            size_v=1
        )
        
        # For each division, create a subfunction
        for j in range(h_divisions):
            # Calculate the actual dimensions of this slice
            start_h = j * self.array_h
            end_h = min((j + 1) * self.array_h, output_dim)
            

            # Create subfunction
            subfunc = SubFunction(1, j, function.k, function.m, function.n, function.op_type)
            subfunc.set_parent(function)

            subfunc.set_shape((1, end_h - start_h))  # Set shape for element-wise operations
            
            if function.op_type == OperationType.GLU:
                # For GLU, there will be two input tensors
                distribution_func1.add_output_tensor(
                    tensor_id=TensorId(i = 1, j = -(j+1),
                                       k=1,
                                       m=function.m,
                                        n=function.n),
                    size_h=end_h - start_h,
                    size_v=1
                )
                subfunc.add_input_tensor(
                    tensor_id=TensorId(i = 1, j = -(j+1),
                                       k=1,
                                       m=function.m,
                                       n=function.n),
                    size_h=end_h - start_h,
                    size_v=1
                )
                distribution_func2.add_output_tensor(
                    tensor_id=TensorId(i = 1, j = -(j+1),
                                       k=2,
                                       m=function.m,
                                        n=function.n),
                    size_h=end_h - start_h,
                    size_v=1
                )
                subfunc.add_input_tensor(
                    tensor_id=TensorId(i = 1, j = -(j+1),
                                       k=2,
                                       m=function.m,
                                       n=function.n),
                    size_h=end_h - start_h,
                    size_v=1
                )
            else:
                # Add output tensor to the distribution function
                distribution_func.add_output_tensor(
                    tensor_id=TensorId(i = 1, j = -(j+1),
                                       k=function.k,
                                       m=function.m,
                                        n=function.n),
                    size_h=end_h - start_h,
                    size_v=1
                )

                subfunc.add_input_tensor(
                    tensor_id=TensorId(i = 1, j = -(j+1),
                                        k=function.k,
                                        m=function.m,
                                        n=function.n),
                    size_h=end_h-start_h,
                    size_v=1
                )
            
            # Add input tensor for the concatenation function
            concat_func.add_input_tensor(
                tensor_id=TensorId(i = 1, j = j+1,
                                   k=function.k,
                                   m=function.m,
                                   n=function.n),
                size_h=end_h - start_h,
                size_v=1
            )
            # Set output tensor
            subfunc.add_output_tensor(
                tensor_id=TensorId(i = 1, j = j+1,
                                   k=function.k,
                                   m=function.m,
                                   n=function.n),
                size_h=1,
                size_v=end_h - start_h
            )
            
            # Add to compiled model
            compiled_model.add_subfunction(subfunc)
        # Add the distribution function to the compiled model
        if function.op_type == OperationType.GLU:
            compiled_model.add_subfunction(distribution_func1)
            compiled_model.add_subfunction(distribution_func2)
        else:
            compiled_model.add_subfunction(distribution_func)
        # Add the concatenation function to the compiled model
        compiled_model.add_subfunction(concat_func)
        # Add pass function to pass to next step
        pass_func = SubFunction(-1, 0, function.k, function.m, function.n, OperationType.PASS)
        # Add input tensor for the pass function
        pass_func.add_input_tensor(
            tensor_id=TensorId(i = 0, j = 0,
                               k=function.k,
                               m=function.m,
                               n=function.n),
            size_h=output_dim,
            size_v=1
        )
        # Add output tensor for the pass function
        pass_func.add_output_tensor(
            tensor_id=TensorId(k=function.k,
                               m=function.m+1,
                               n=function.n),
            size_h=output_dim,
            size_v=1
        )
        # Add the pass function to the compiled model
        compiled_model.add_subfunction(pass_func)

def create_glu_ffn_model(hidden_dim: int, ffn_dim: int, layer_idx: int = 1) -> Model:
    """
    Create a model for a GLU-based FFN layer
    
    Args:
        hidden_dim: Model dimension (hidden state size)
        ffn_dim: FFN dimension (intermediate size)
        layer_idx: Decoder layer index
        
    Returns:
        Model object representing the FFN layer
    """
    model = Model()
    
    # Input tensor (from previous layer)
    input_tensor_id = TensorId(k=0, m=0, n=layer_idx-1)
    
    # 1. Up projection (k=1, m=1)
    up_proj = Function(k=1, m=1, n=layer_idx, op_type=OperationType.MVM)
    up_proj.set_shape((hidden_dim, ffn_dim))
    model.add_function(up_proj)
    
    # 2. Gate projection (k=2, m=1)
    gate_proj = Function(k=2, m=1, n=layer_idx, op_type=OperationType.MVM)
    gate_proj.set_shape((hidden_dim, ffn_dim))
    model.add_function(gate_proj)
    
    # 3. Trivial copy for up_proj (k=1, m=2)
    up_copy = Function(k=1, m=2, n=layer_idx, op_type=OperationType.TRIVIAL_COPY)
    up_copy.set_shape((1, ffn_dim))
    model.add_function(up_copy)
    
    # 4. Activation for gate_proj (k=2, m=2)
    activation = Function(k=2, m=2, n=layer_idx, op_type=OperationType.ACTIVATION)
    activation.set_shape((1, ffn_dim))
    model.add_function(activation)
    
    # 5. GLU operation (k=1, m=3)
    glu = Function(k=1, m=3, n=layer_idx, op_type=OperationType.GLU)
    glu.set_shape((1, ffn_dim))
    model.add_function(glu)
    
    # 6. Down projection (k=1, m=4)
    down_proj = Function(k=1, m=4, n=layer_idx, op_type=OperationType.MVM)
    down_proj.set_shape((ffn_dim, hidden_dim))
    model.add_function(down_proj)
    
    return model

#=======================visualization=========================
import graphviz
from typing import Dict, Set, List


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