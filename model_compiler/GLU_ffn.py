from model_compiler.utils import OperationType, TensorId, TensorWithSize, Function, SubFunction, Model, CompiledModel


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
    
    # Standard coordinate system for this model
    base_coords = {'n': layer_idx}
    
    # Input tensor (from previous layer)
    input_tensor_id = TensorId(k=0, m=0, n=layer_idx-1)
    
    # 1. Up projection (k=1, m=1)
    up_proj = Function(op_type=OperationType.MVM, k=1, m=1, n=layer_idx)
    up_proj.set_shape((hidden_dim, ffn_dim))
    model.add_function(up_proj)
    
    # 2. Gate projection (k=2, m=1)
    gate_proj = Function(op_type=OperationType.MVM, k=2, m=1, n=layer_idx)
    gate_proj.set_shape((hidden_dim, ffn_dim))
    model.add_function(gate_proj)
    
    # 3. Trivial copy for up_proj (k=1, m=2)
    up_copy = Function(op_type=OperationType.TRIVIAL_COPY, k=1, m=2, n=layer_idx)
    up_copy.set_shape((1, ffn_dim))
    model.add_function(up_copy)
    
    # 4. Activation for gate_proj (k=2, m=2)
    activation = Function(op_type=OperationType.ACTIVATION, k=2, m=2, n=layer_idx)
    activation.set_shape((1, ffn_dim))
    model.add_function(activation)
    
    # 5. GLU operation (k=1, m=3)
    glu = Function(op_type=OperationType.GLU, k=1, m=3, n=layer_idx)
    glu.set_shape((1, ffn_dim))
    model.add_function(glu)
    
    # 6. Down projection (k=1, m=4)
    down_proj = Function(op_type=OperationType.MVM, k=1, m=4, n=layer_idx)
    down_proj.set_shape((ffn_dim, hidden_dim))
    model.add_function(down_proj)
    
    return model
