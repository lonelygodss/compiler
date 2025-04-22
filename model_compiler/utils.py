from enum import Enum
from typing import List, Dict, Tuple, Set, Optional, Any, Union, Callable

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
    """Flexible identifier for tensors in the model"""
    def __init__(self, **kwargs):
        self.coords = kwargs  # Store all coordinates as a dictionary
    
    @classmethod
    def create(cls, **kwargs):
        """Factory method to create a TensorId with specific coordinates"""
        return cls(**kwargs)
    
    def with_coords(self, **kwargs):
        """Create a new TensorId with updated coordinates"""
        new_coords = self.coords.copy()
        new_coords.update(kwargs)
        return TensorId(**new_coords)
    
    def get(self, key, default=None):
        """Get a coordinate value by key"""
        return self.coords.get(key, default)
    
    def __str__(self):
        coords_str = ", ".join(f"{k}={v}" for k, v in sorted(self.coords.items()))
        return f"Tensor({coords_str})"
    
    def __eq__(self, other):
        if not isinstance(other, TensorId):
            return False
        return self.coords == other.coords
    
    def __hash__(self):
        return hash(tuple(sorted(self.coords.items())))


class TensorWithSize:
    """Represents a instance of a tensor """
    def __init__(self, tensor_id: TensorId, **size_params):
        self.tensor_id = tensor_id
        self.size_params = size_params  # Flexible size parameters
    
    def __str__(self):
        size_info = ""
        if self.size_params:
            size_info = f", size={self.size_params}"
        return f"{self.tensor_id}{size_info}"


class Function:
    """High-level function in the model (before division)"""
    def __init__(self, op_type: OperationType, **coords):
        self.op_type = op_type
        self.coords = coords  # Store all coordinates as a dictionary
        self.shape = None     # For operations with shape requirements
        self.metadata = {}    # Additional metadata for the function
    
    def set_shape(self, shape: Tuple):
        """Set the shape for operations"""
        self.shape = shape
        
    def set_metadata(self, key, value):
        """Set additional metadata"""
        self.metadata[key] = value
        
    def get_metadata(self, key, default=None):
        """Get metadata value by key"""
        return self.metadata.get(key, default)
        
    def __str__(self):
        coords_str = ", ".join(f"{k}={v}" for k, v in sorted(self.coords.items()))
        shape_info = f", shape={self.shape}" if self.shape else ""
        return f"Function({coords_str}, op={self.op_type.value}{shape_info})"


class SubFunction:
    """Sub-function after division according to hardware constraints"""
    def __init__(self, op_type: OperationType, **coords):
        self.op_type = op_type
        self.coords = coords  # Store all coordinates as a dictionary
        self.input_tensors: List[TensorWithSize] = []
        self.output_tensors: List[TensorWithSize] = []
        self.shape = None     # For operations with shape requirements
        self.parent_function = None  # Reference to the parent Function
        self.metadata = {}    # Additional metadata for the subfunction
        
    def set_parent(self, parent: Function):
        """Set the parent function this subfunction was derived from"""
        self.parent_function = parent
        
    def set_shape(self, shape: Tuple):
        """Set the shape for operations"""
        self.shape = shape
        
    def set_metadata(self, key, value):
        """Set additional metadata"""
        self.metadata[key] = value
        
    def get_metadata(self, key, default=None):
        """Get metadata value by key"""
        return self.metadata.get(key, default)
        
    def add_input_tensor(self, tensor_id: TensorId, **size_params):
        """Add an input tensor dependency"""
        self.input_tensors.append(TensorWithSize(tensor_id, **size_params))
        
    def add_output_tensor(self, tensor_id: TensorId, **size_params):
        """Register output tensor with the function coordinates"""
        self.output_tensors.append(TensorWithSize(tensor_id, **size_params))
        
    def __str__(self):
        coords_str = ", ".join(f"{k}={v}" for k, v in sorted(self.coords.items()))
        inputs = ', '.join(str(ts.tensor_id) for ts in self.input_tensors)
        outputs = ', '.join(str(ts.tensor_id) for ts in self.output_tensors)
        shape_info = f", shape={self.shape}" if self.shape else ""
        return f"SubFunction({coords_str}, op={self.op_type.value}, inputs=[{inputs}], outputs=[{outputs}]{shape_info})"


class Model:
    """Holds the complete model description"""
    def __init__(self):
        self.functions: List[Function] = []
        self.metadata = {}    # Additional metadata for the model
        
    def add_function(self, function: Function):
        """Add a function to the model"""
        self.functions.append(function)
        
    def get_functions_by_coords(self, **coords):
        """Get functions matching specific coordinates"""
        results = []
        for func in self.functions:
            match = True
            for key, value in coords.items():
                if func.coords.get(key) != value:
                    match = False
                    break
            if match:
                results.append(func)
        return results
    
    def set_metadata(self, key, value):
        """Set additional metadata"""
        self.metadata[key] = value
        
    def get_metadata(self, key, default=None):
        """Get metadata value by key"""
        return self.metadata.get(key, default)
    
    def __str__(self):
        return '\n'.join(str(func) for func in self.functions)


class CompiledModel:
    """Holds the divided model with subfunctions"""
    def __init__(self):
        self.subfunctions: List[SubFunction] = []
        self.dependency_graph: Dict[TensorId, List[SubFunction]] = {}
        self.metadata = {}    # Additional metadata for the compiled model
        
    def add_subfunction(self, subfunction: SubFunction):
        """Add a subfunction to the compiled model"""
        self.subfunctions.append(subfunction)
        
    def get_subfunctions_by_coords(self, **coords):
        """Get subfunctions matching specific coordinates"""
        results = []
        for subfunc in self.subfunctions:
            match = True
            for key, value in coords.items():
                if subfunc.coords.get(key) != value:
                    match = False
                    break
            if match:
                results.append(subfunc)
        return results
        
    def build_dependency_graph(self):
        """Build a graph showing which subfunctions depend on which tensors"""
        # Reset the graph
        self.dependency_graph = {}
        
        # Add all output tensors to the graph
        for subfunc in self.subfunctions:
            for output_tensor in subfunc.output_tensors:
                if output_tensor.tensor_id not in self.dependency_graph:
                    self.dependency_graph[output_tensor.tensor_id] = []
        
        # For each subfunction, add its dependencies
        for subfunc in self.subfunctions:
            for input_tensor in subfunc.input_tensors:
                if input_tensor.tensor_id in self.dependency_graph:
                    self.dependency_graph[input_tensor.tensor_id].append(subfunc)
    
    def set_metadata(self, key, value):
        """Set additional metadata"""
        self.metadata[key] = value
        
    def get_metadata(self, key, default=None):
        """Get metadata value by key"""
        return self.metadata.get(key, default)
        
    def __str__(self):
        return '\n'.join(str(subfunc) for subfunc in self.subfunctions)