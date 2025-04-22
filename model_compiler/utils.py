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