import keras
import numpy as np
import matplotlib.pyplot as plt

from resolv_ml.utilities.schedulers import get_scheduler

if __name__ == '__main__':
    total_steps = 40000
    resolution = 5000
    x = np.linspace(0, total_steps, num=resolution)

    # KLD - Beta
    kld_schedule_type = "exponential"
    kld_schedule_config = {
        "rate": 0.9999,
        "min_value": 0.0,
        "max_value": 0.2,
        "decay": False
    }
    kld_scheduler = get_scheduler(kld_schedule_type, kld_schedule_config)
    y_kld_scheduler = [kld_scheduler(step) for step in x]

    # Attribute Regularizer - Beta
    attr_reg_schedule_type = "constant"
    attr_reg_schedule_config = {
        "value": 1
    }
    attr_reg_scheduler = get_scheduler(attr_reg_schedule_type, attr_reg_schedule_config)
    y_attr_reg_scheduler = [attr_reg_scheduler(step) for step in x]

    # Sampling probability
    sampling_prob_schedule_type = "sigmoid"
    sampling_prob_schedule_config = {
        "rate": 2000,
        "min_value": 0.0,
        "max_value": 1.0,
        "decay": False
    }
    sampling_prob_scheduler = get_scheduler(sampling_prob_schedule_type, sampling_prob_schedule_config)
    y_sampling_prob_scheduler = [sampling_prob_scheduler(step) for step in x]

    # Learning rate
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_rate=0.9999,
        decay_steps=1,
        staircase=False
    )
    y_lr_scheduler = [lr_scheduler(step) for step in x]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, y_kld_scheduler, label="$\\beta$-KLD", color='blue')
    plt.plot(x, y_attr_reg_scheduler, label="$\\gamma$-Attr", color='red')
    plt.plot(x, y_sampling_prob_scheduler, label="P-Sampling", color='green')
    plt.axhline(max(y_kld_scheduler), color='purple', linestyle='--', linewidth=0.8,
                label=f'max(kld)={max(y_kld_scheduler):.4f}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(1, 2, 2)
    plt.plot(x, y_lr_scheduler, label="LR", color='purple')
    plt.axhline(min(y_lr_scheduler), color='blue', linestyle='--', linewidth=0.8,
                label=f'min(lr)={min(y_lr_scheduler):.8f}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()
