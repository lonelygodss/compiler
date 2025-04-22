import compiler.bulkversion.model_compiler_bulkversion as mc


def main():
    # Example usage
    # Define model parameters
    hidden_dim = 4096  # Model dimension (e.g., for Llama 7B)
    ffn_dim = 11008    # FFN dimension (e.g., for Llama 7B)
    layer_idx = 1      # First decoder layer
    
    # Define hardware constraints
    array_h = 2048      # Horizontal size of CIM array
    array_v = 2048      # Vertical size of CIM array
    
    # Create model
    model = mc.create_glu_ffn_model(hidden_dim, ffn_dim, layer_idx)
    print("Original Model:")
    print(model)
    print("\n" + "="*80 + "\n")
    
    # Compile model
    compiler = mc.Compiler(array_h, array_v)
    compiled_model = compiler.divide_model(model)
    
    print("Compiled Model:")
    print(f"Total subfunctions: {len(compiled_model.subfunctions)}")
    
    # Print some statistics
    op_counts = {}
    for subfunc in compiled_model.subfunctions:
        op_type = subfunc.op_type.value
        if op_type not in op_counts:
            op_counts[op_type] = 0
        op_counts[op_type] += 1
    
    print("\nOperation counts:")
    for op_type, count in op_counts.items():
        print(f"  {op_type}: {count}")
    
    # Print a sample of subfunctions
    # print("\nSample subfunctions:")
    # for i, subfunc in enumerate(compiled_model.subfunctions[:]):
    #      print(f"  {i+1}. {subfunc}")
    
    # if len(compiled_model.subfunctions) > 50:
    #     print(f"  ... and {len(compiled_model.subfunctions) - 50} more")

    # Visualize the compiled model with shorter labels
    mc.visualize_compiled_model(compiled_model, "ffn_compiled_model")
    
    # Alternative layered visualization with shorter labels
    mc.visualize_compiled_model_layered(compiled_model, "ffn_compiled_model_layered")
    
    # Simplified visualization focusing on dataflow
    mc.visualize_compiled_model_simple(compiled_model, "ffn_compiled_model_simple")

    # Parse and analyze the compute graph
    connection_info = mc.parse_compute_graph(compiled_model)    
    # Save the compute graph
    mc.save_compute_graph(connection_info, "ffn_compute_graph.json")
    
    # Visualize the compute graph
    mc.visualize_compute_graph_graphviz(connection_info, "ffn_compute_graph_graphviz")
    
    # Analyze the compute graph
    analysis = mc.analyze_compute_graph(connection_info)
    print("\nCompute Graph Analysis:")
    for key, value in analysis.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()