from training.glfc.glfc import init_glfc_components

def load_training_method(TrainingArgs, feature_extractor, train_dataset, encode_model):
    # Initialize components based on method
    if TrainingArgs.method == "GLFC":
        global_model, trainers = init_glfc_components(
            TrainingArgs, feature_extractor, train_dataset, encode_model
        )
        return global_model, trainers
    else:
        raise NotImplementedError(f"Method {TrainingArgs.method} not implemented")