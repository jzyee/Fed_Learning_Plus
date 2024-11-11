from encoder.LeNet import LeNet

def load_encoder(encoder_args, training_args):
    if encoder_args.encoder_name == 'LeNet':
        total_num_classes = training_args.tasks_global * training_args.task_size
        return LeNet(num_classes=total_num_classes)
    else:
        raise ValueError(f"Encoder {encoder_args.encoder_name} not supported")
    