# Basic compiler with function derived concat, distribution and addition

from model_compiler.utils import OperationType,TensorId,TensorWithSize,Function,SubFunction,Model,CompiledModel


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