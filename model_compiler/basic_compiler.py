# Basic compiler with function derived concat, distribution and addition

from model_compiler.utils import OperationType, TensorId, TensorWithSize, Function, SubFunction, Model, CompiledModel


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
    
    def _create_tensor_id(self, base_coords, **override_coords):
        """Helper to create a tensor ID with consistent naming"""
        coords = base_coords.copy()
        coords.update(override_coords)
        return TensorId(**coords)
    
    def _create_subfunction(self, base_coords, op_type, **override_coords):
        """Helper to create a subfunction with consistent naming"""
        coords = base_coords.copy()
        coords.update(override_coords)
        return SubFunction(op_type=op_type, **coords)
    
    def _add_pass_function(self, function, compiled_model, output_size, output_tensor_id=None):
        """Helper to create a pass function to the next step"""
        base_coords = function.coords.copy()
        
        # Create pass function
        pass_func = self._create_subfunction(base_coords, OperationType.PASS, i=-1, j=0)
        
        # Set up input tensor - from concat result
        input_tensor_id = self._create_tensor_id(base_coords, i=0, j=0)            
        pass_func.add_input_tensor(input_tensor_id, size_h=output_size, size_v=1)
        
        # Set up output tensor - to next step
        if output_tensor_id is None:
            # Default to next step in sequence
            m_key = 'm'  # Assuming 'm' is used for sequence step
            next_m = base_coords.get(m_key, 0) + 1
            output_tensor_id = self._create_tensor_id({k: v for k, v in base_coords.items() if k != m_key}, **{m_key: next_m})
            
        pass_func.add_output_tensor(output_tensor_id, size_h=output_size, size_v=1)
        compiled_model.add_subfunction(pass_func)
    
    def _divide_mvm(self, function: Function, compiled_model: CompiledModel):
        """Divide an MVM function into subfunctions based on array size constraints"""
        if not function.shape:
            raise ValueError(f"Function {function} has no shape defined, required for MVM division")
            
        input_dim, output_dim = function.shape
        base_coords = function.coords.copy()
        
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
                compute_i = i + 1  # Base index for MVM computation
                compute_j = j + 1  # Base index for MVM computation
                subfunc = self._create_subfunction(
                    base_coords, 
                    OperationType.MVM, 
                    i=compute_i, 
                    j=compute_j
                )
                subfunc.set_shape((end_v - start_v, end_h - start_h))
                subfunc.set_parent(function)
                
                # Input tensor ID for this slice
                input_tensor_id = self._create_tensor_id(base_coords, i=-compute_i, j=-compute_j)
                subfunc.add_input_tensor(input_tensor_id, size_h=end_v - start_v, size_v=1)
                
                # Output tensor ID for compute result
                output_tensor_id = self._create_tensor_id(base_coords, i=compute_i, j=compute_j)
                subfunc.add_output_tensor(output_tensor_id, size_h=end_h - start_h, size_v=1)
                
                # Add to compiled model
                compiled_model.add_subfunction(subfunc)

        # Create distribution function
        distribution_func = self._create_subfunction(
            base_coords, 
            OperationType.DISTRIBUTE, 
            i=0, 
            j=-1
        )
        
        # Get standard input key from metadata or use default (assuming 'k' for parallel paths)
        k_key = 'k'
        default_input_k = base_coords.get(k_key, 1)
        
        # Input tensor for distribution (from previous layer/step)
        input_coords = {k: v for k, v in base_coords.items()}
        input_coords[k_key] = default_input_k
        input_tensor_id = TensorId(**input_coords)
        distribution_func.add_input_tensor(input_tensor_id, size_h=input_dim, size_v=1)
        
        # Add output tensors for each compute function input
        for i in range(v_divisions):
            for j in range(h_divisions):
                compute_i = i + 1
                compute_j = j + 1
                start_v = i * self.array_v
                end_v = min((i + 1) * self.array_v, input_dim)
                
                # Each distribution output connects to a compute function input
                output_tensor_id = self._create_tensor_id(base_coords, i=-compute_i, j=-compute_j)
                distribution_func.add_output_tensor(output_tensor_id, size_h=end_v - start_v, size_v=1)
        
        # Add distribution function to model
        compiled_model.add_subfunction(distribution_func)
        
        # Create concat function for final results
        concat_func = self._create_subfunction(base_coords, OperationType.CONCAT, i=0, j=0)
        concat_func.add_output_tensor(
            self._create_tensor_id(base_coords, i=0, j=0), size_h=output_dim, size_v=1
        )
        
        # For each column of divisions, create an add function to combine vertical slices
        for j in range(h_divisions):
            start_h = j * self.array_h
            end_h = min((j + 1) * self.array_h, output_dim)
            add_j = j + 1  # Base index for addition operations
            
            # Create addition function
            add_func = self._create_subfunction(
                base_coords, 
                OperationType.ADD, 
                i=0, 
                j=add_j
            )
            add_func.set_shape((1, end_h - start_h))
            
            # Set output tensor for the addition function
            add_output_tensor_id = self._create_tensor_id(base_coords, i=0, j=add_j)
            add_func.add_output_tensor(add_output_tensor_id, size_h=end_h - start_h, size_v=1)
            
            # Add each compute result as input to the addition function
            for i in range(v_divisions):
                compute_i = i + 1
                compute_j = j + 1
                
                # Link compute output to add input
                add_input_tensor_id = self._create_tensor_id(base_coords, i=compute_i, j=compute_j)
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
        base_coords = function.coords.copy()
        
        # Determine the output dimension
        output_dim = None
        
        if function.shape:
            _, output_dim = function.shape  # This is a simplification - in reality, we'd need to know the exact output shape
        else:
            # Try to infer from the first input tensor or use default
            output_dim = 1024  # Default value, should be determined from model architecture
        
        # Calculate number of divisions needed (only in horizontal dimension for elementwise ops)
        h_divisions = (output_dim + self.array_h - 1) // self.array_h

        # For GLU, we need two distribution functions
        if function.op_type == OperationType.GLU:
            distribution_funcs = []
            for k_idx in [1, 2]:  # GLU has two inputs
                k_key = 'k'  # Assuming 'k' is used for parallel paths
                dist_coords = base_coords.copy()
                dist_coords[k_key] = k_idx  # Update k value for this distribution function
                
                dist_func = self._create_subfunction(
                    dist_coords,
                    OperationType.DISTRIBUTE,
                    i=0,
                    j=-1
                )
                
                # Input tensor for this distribution function
                input_tensor_id = TensorId(**dist_coords)
                dist_func.add_input_tensor(input_tensor_id, size_h=output_dim, size_v=1)
                distribution_funcs.append(dist_func)
        else:
            # Create single distribution function for non-GLU operations
            distribution_func = self._create_subfunction(
                base_coords,
                OperationType.DISTRIBUTE,
                i=0,
                j=-1
            )
            # Add input for distribution
            input_tensor_id = TensorId(**base_coords)
            distribution_func.add_input_tensor(input_tensor_id, size_h=output_dim, size_v=1)
            distribution_funcs = [distribution_func]
        
        # Create concat function
        concat_func = self._create_subfunction(
            base_coords,
            OperationType.CONCAT,
            i=0,
            j=0
        )
        # Add output tensor for concat
        concat_func.add_output_tensor(
            self._create_tensor_id(base_coords, i=0, j=0),
            size_h=output_dim,
            size_v=1
        )
        
        # Create compute subfunctions for each division
        element_idx = 1  # Index for elementwise operations
        for j in range(h_divisions):
            # Calculate slice dimensions
            start_h = j * self.array_h
            end_h = min((j + 1) * self.array_h, output_dim)
            slice_size = end_h - start_h
            
            # Create compute subfunction
            subfunc = self._create_subfunction(
                base_coords,
                function.op_type,
                i=element_idx,
                j=j+1
            )
            subfunc.set_shape((1, slice_size))
            subfunc.set_parent(function)
            
            # Handle different input configurations for GLU vs other operations
            if function.op_type == OperationType.GLU:
                # Create unique tensor IDs for each GLU input
                k_key = 'k'
                for k_idx, dist_func in enumerate(distribution_funcs, 1):
                    input_coords = base_coords.copy()
                    input_coords[k_key] = k_idx
                    input_coords['i'] = element_idx
                    input_coords['j'] = -(j+1)
                    
                    input_tensor_id = TensorId(**input_coords)
                    
                    # Add corresponding output to distribution function
                    dist_func.add_output_tensor(input_tensor_id, size_h=slice_size, size_v=1)
                    
                    # Add input to compute function
                    subfunc.add_input_tensor(input_tensor_id, size_h=slice_size, size_v=1)
            else:
                # Single input for non-GLU operations
                input_tensor_id = self._create_tensor_id(base_coords, i=element_idx, j=-(j+1))
                
                # Add corresponding output to distribution function
                distribution_funcs[0].add_output_tensor(input_tensor_id, size_h=slice_size, size_v=1)
                
                # Add input to compute function
                subfunc.add_input_tensor(input_tensor_id, size_h=slice_size, size_v=1)
            
            # Set output tensor for compute function
            output_tensor_id = self._create_tensor_id(base_coords, i=element_idx, j=j+1)
            subfunc.add_output_tensor(output_tensor_id, size_h=slice_size, size_v=1)
            
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