# Basic compiler with function derived concat, distribution and addition

from model_compiler.utils import OperationType,TensorId,TensorWithSize,Function,SubFunction,Model,CompiledModel


class Compiler:
    """Compiler that divides the model according to hardware constraints"""
    # Constants to replace hardcoded indices
    MVM_COMPUTE_IDX = 1        # Base index for MVM computation subfunctions
    DISTRIBUTE_IDX = 0         # Horizontal index for distribution
    DISTRIBUTE_VERT_IDX = -1   # Vertical index for distribution
    CONCAT_IDX = 0             # Index for concatenation operations
    ADD_IDX = 0                # Base horizontal index for addition operations
    PASS_IDX = -1              # Index for pass operations
    
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
    
    def _create_tensor_id(self, i, j, function, k_override=None):
        """Helper to create a tensor ID with consistent naming"""
        k = k_override if k_override is not None else function.k
        return TensorId(i=i, j=j, k=k, m=function.m, n=function.n)
    
    def _create_subfunction(self, i, j, function, op_type, shape=None):
        """Helper to create a subfunction with consistent naming"""
        subfunc = SubFunction(i, j, function.k, function.m, function.n, op_type)
        if shape:
            subfunc.set_shape(shape)
        if op_type == function.op_type:
            subfunc.set_parent(function)
        return subfunc
    
    def _add_pass_function(self, function, compiled_model, input_size, output_tensor_id=None):
        """Helper to create a pass function to the next step"""
        pass_func = self._create_subfunction(self.PASS_IDX, 0, function, OperationType.PASS)
        
        # Set up input tensor - from concat result
        input_tensor_id = self._create_tensor_id(0, 0, function)
        pass_func.add_input_tensor(input_tensor_id, size_h=input_size, size_v=1)
        
        # Set up output tensor - to next step
        if output_tensor_id is None:
            # Default to next step in sequence
            output_tensor_id = TensorId(k=function.k, m=function.m+1, n=function.n)
        
        pass_func.add_output_tensor(output_tensor_id, size_h=input_size, size_v=1)
        compiled_model.add_subfunction(pass_func)
    
    def _divide_mvm(self, function: Function, compiled_model: CompiledModel):
        """Divide an MVM function into subfunctions based on array size constraints"""
        if not function.shape:
            raise ValueError(f"Function {function} has no shape defined, required for MVM division")
            
        input_dim, output_dim = function.shape
        
        # Calculate number of divisions needed
        h_divisions = (output_dim + self.array_h - 1) // self.array_h
        v_divisions = (input_dim + self.array_v - 1) // self.array_v
        
        # Create compute subfunctions for each division
        for i in range(v_divisions):
            for j in range(h_divisions):
                # Calculate the actual dimensions of this submatrix
                start_h = j * self.array_h
                end_h = min((j + 1) * self.array_h, output_dim)
                start_v = i * self.array_v
                end_v = min((i + 1) * self.array_v, input_dim)
                
                # Create subfunction
                compute_i = i + self.MVM_COMPUTE_IDX
                compute_j = j + self.MVM_COMPUTE_IDX
                subfunc = self._create_subfunction(compute_i, compute_j, function, OperationType.MVM, 
                                                  (end_v - start_v, end_h - start_h))
                
                # Input tensor ID for this slice
                input_tensor_id = self._create_tensor_id(-compute_i, -compute_j, function)
                subfunc.add_input_tensor(input_tensor_id, size_v=1, size_h=end_v - start_v)
                
                # Output tensor ID for compute result
                output_tensor_id = self._create_tensor_id(compute_i, compute_j, function)
                subfunc.add_output_tensor(output_tensor_id, size_v=1, size_h=end_h - start_h)

                # Add to compiled model
                compiled_model.add_subfunction(subfunc)

        # Create distribution function
        distribution_func = self._create_subfunction(self.DISTRIBUTE_IDX, self.DISTRIBUTE_VERT_IDX, 
                                                   function, OperationType.DISTRIBUTE)
        
        # Input tensor for distribution (from previous layer/step)
        input_tensor_id = TensorId(k=1, m=function.m, n=function.n)  # Using k=1 as default
        distribution_func.add_input_tensor(input_tensor_id, size_h=input_dim, size_v=1)
        
        # Add output tensors for each compute function input
        for i in range(v_divisions):
            for j in range(h_divisions):
                compute_i = i + self.MVM_COMPUTE_IDX
                compute_j = j + self.MVM_COMPUTE_IDX
                start_v = i * self.array_v
                end_v = min((i + 1) * self.array_v, input_dim)
                
                # Each distribution output connects to a compute function input
                output_tensor_id = self._create_tensor_id(-compute_i, -compute_j, function)
                distribution_func.add_output_tensor(output_tensor_id, size_h=end_v - start_v, size_v=1)
        
        # Add distribution function to model
        compiled_model.add_subfunction(distribution_func)

        # Create concat function for final results
        concat_func = self._create_subfunction(self.CONCAT_IDX, self.CONCAT_IDX, function, OperationType.CONCAT)
        concat_func.add_output_tensor(
            self._create_tensor_id(0, 0, function), size_h=output_dim, size_v=1
        )
        
        # For each column of divisions, create an add function to combine vertical slices
        for j in range(h_divisions):
            start_h = j * self.array_h
            end_h = min((j + 1) * self.array_h, output_dim)
            add_j = j + 1  # Base index for addition operations
            
            # Create addition function
            add_func = self._create_subfunction(self.ADD_IDX, add_j, function, OperationType.ADD, 
                                              (1, end_h - start_h))
            
            # Set output tensor for the addition function
            add_output_tensor_id = self._create_tensor_id(0, add_j, function)
            add_func.add_output_tensor(add_output_tensor_id, size_h=end_h - start_h, size_v=1)
            
            # Add each compute result as input to the addition function
            for i in range(v_divisions):
                compute_i = i + self.MVM_COMPUTE_IDX
                compute_j = j + self.MVM_COMPUTE_IDX
                
                # Link compute output to add input
                add_input_tensor_id = self._create_tensor_id(compute_i, compute_j, function)
                add_func.add_input_tensor(add_input_tensor_id, size_h=end_h - start_h, size_v=1)
            
            # Add the addition function to the model
            compiled_model.add_subfunction(add_func)
            
            # Link addition output to concatenation input
            concat_func.add_input_tensor(add_output_tensor_id, size_h=end_h - start_h, size_v=1)
        
        # Add concat function to model
        compiled_model.add_subfunction(concat_func)
        
        # Add pass function to next step
        self._add_pass_function(function, compiled_model, output_dim)

    def _divide_elementwise(self, function: Function, compiled_model: CompiledModel):
        """Divide element-wise operations (activation, GLU, etc.) into subfunctions"""
        # Determine the output dimension
        output_dim = None
        
        if function.shape:
            _, output_dim = function.shape
        else:
            # Try to infer from the first input tensor or use default
            output_dim = 1024  # Default value, should be determined from model architecture
        
        # Calculate number of divisions needed (only in horizontal dimension for elementwise ops)
        h_divisions = (output_dim + self.array_h - 1) // self.array_h

        # For GLU, we need two distribution functions
        if function.op_type == OperationType.GLU:
            distribution_funcs = []
            for k_idx in [1, 2]:  # GLU has two inputs
                dist_func = self._create_subfunction(self.DISTRIBUTE_IDX, self.DISTRIBUTE_VERT_IDX, 
                                                    function, OperationType.DISTRIBUTE)
                dist_func.k = k_idx  # Override k value for GLU inputs
                
                # Input tensor for this distribution function
                input_tensor_id = TensorId(k=k_idx, m=function.m, n=function.n)
                dist_func.add_input_tensor(input_tensor_id, size_h=output_dim, size_v=1)
                distribution_funcs.append(dist_func)
        else:
            # Create single distribution function for non-GLU operations
            distribution_func = self._create_subfunction(self.DISTRIBUTE_IDX, self.DISTRIBUTE_VERT_IDX, 
                                                       function, OperationType.DISTRIBUTE)
            # Add input for distribution
            input_tensor_id = TensorId(k=function.k, m=function.m, n=function.n)
            distribution_func.add_input_tensor(input_tensor_id, size_h=output_dim, size_v=1)
            distribution_funcs = [distribution_func]
        
        # Create concat function
        concat_func = self._create_subfunction(self.CONCAT_IDX, self.CONCAT_IDX, 
                                             function, OperationType.CONCAT)
        # Add output tensor for concat
        concat_func.add_output_tensor(
            self._create_tensor_id(0, 0, function), size_h=output_dim, size_v=1
        )
        
        # Create compute subfunctions for each division
        element_idx = 1  # Index for elementwise operations
        for j in range(h_divisions):
            # Calculate slice dimensions
            start_h = j * self.array_h
            end_h = min((j + 1) * self.array_h, output_dim)
            slice_size = end_h - start_h
            
            # Create compute subfunction
            subfunc = self._create_subfunction(element_idx, j, function, function.op_type, 
                                              (1, slice_size))
            
            # Handle different input configurations for GLU vs other operations
            if function.op_type == OperationType.GLU:
                for k_idx, dist_func in enumerate(distribution_funcs, 1):
                    # Create unique tensor IDs for each GLU input
                    input_tensor_id = self._create_tensor_id(element_idx, -(j+1), function, k_override=k_idx)
                    subfunc.add_input_tensor(input_tensor_id, size_h=slice_size, size_v=1)
                    
                    # Add corresponding output to distribution function
                    dist_func.add_output_tensor(input_tensor_id, size_h=slice_size, size_v=1)
            else:
                # Single input for non-GLU operations
                input_tensor_id = self._create_tensor_id(element_idx, -(j+1), function)
                subfunc.add_input_tensor(input_tensor_id, size_h=slice_size, size_v=1)
                
                # Add corresponding output to distribution function
                distribution_funcs[0].add_output_tensor(input_tensor_id, size_h=slice_size, size_v=1)
            
            # Set output tensor for compute function
            output_tensor_id = self._create_tensor_id(element_idx, j+1, function)
            subfunc.add_output_tensor(output_tensor_id, size_h=1, size_v=slice_size)
            
            # Link compute output to concat input
            concat_func.add_input_tensor(output_tensor_id, size_h=slice_size, size_v=1)
            
            # Add to compiled model
            compiled_model.add_subfunction(subfunc)
        
        # Add distribution functions to model
        for dist_func in distribution_funcs:
            compiled_model.add_subfunction(dist_func)
        
        # Add concat function to model
        compiled_model.add_subfunction(concat_func)
        
        # Add pass function to next step
        self._add_pass_function(function, compiled_model, output_dim)